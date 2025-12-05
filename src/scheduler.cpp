#include "scheduler.h"
#include <algorithm>
#include <cstring>

// 类型定义
typedef int cudnnStatus_t;
typedef int cublasStatus_t;

namespace orion {

// 全局调度器实例
Scheduler g_scheduler;

// ============================================================================
// 外部函数声明 (在对应的 intercept.cpp 中定义)
// ============================================================================

extern cudaError_t execute_cuda_operation(OperationPtr op, cudaStream_t scheduler_stream);
extern cudnnStatus_t execute_cudnn_operation(OperationPtr op, cudaStream_t scheduler_stream);
extern cublasStatus_t execute_cublas_operation(OperationPtr op, cudaStream_t scheduler_stream);

// ============================================================================
// Scheduler 实现
// ============================================================================

Scheduler::Scheduler() 
    : running_(false)
    , initialized_(false)
    , hp_stream_(nullptr)
    , hp_task_running_(false)
    , current_hp_op_(nullptr)
    , active_be_count_(0)
    , cumulative_be_duration_ms_(0.0f)
    , profile_table_(nullptr)
    , num_clients_(0) {
}

Scheduler::~Scheduler() {
    stop();
    join();
    destroy_streams();
}

bool Scheduler::init(int num_clients, const SchedulerConfig& config) {
    if (initialized_.load()) {
        LOG_WARN("Scheduler already initialized");
        return true;
    }
    
    num_clients_ = num_clients;
    config_ = config;
    
    // 获取 GPU SM 数量
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, config_.device_id);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to get device properties: %s", cudaGetErrorString(err));
        return false;
    }
    config_.num_sms = prop.multiProcessorCount;
    LOG_INFO("GPU has %d SMs", config_.num_sms);
    
    // 创建 CUDA streams
    if (!create_streams()) {
        return false;
    }
    
    // 预分配空间
    outstanding_kernels_.reserve(64);
    threads_.reserve(num_clients);
    
    initialized_.store(true);
    LOG_INFO("Scheduler initialized with %d clients (multi-threaded)", num_clients);
    LOG_INFO("  SM threshold: %d (%.1f%%)", 
             config_.get_sm_threshold(), config_.sm_threshold_ratio * 100);
    LOG_INFO("  Duration threshold: %.3f ms (%.1f%% of HP latency)", 
             config_.get_dur_threshold_ms(), config_.dur_threshold_ratio * 100);
    
    return true;
}

bool Scheduler::create_streams() {
    // 创建高优先级 stream
    int lowest_priority, highest_priority;
    cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);
    
    cudaError_t err = cudaStreamCreateWithPriority(&hp_stream_, 
                                                    cudaStreamNonBlocking, 
                                                    highest_priority);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to create HP stream: %s", cudaGetErrorString(err));
        return false;
    }
    LOG_DEBUG("Created HP stream with priority %d", highest_priority);
    
    // 为每个 BE client 创建一个 stream
    be_streams_.resize(num_clients_ - 1);  // client 0 是 HP
    for (int i = 0; i < num_clients_ - 1; i++) {
        err = cudaStreamCreateWithPriority(&be_streams_[i], 
                                           cudaStreamNonBlocking, 
                                           lowest_priority);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to create BE stream %d: %s", i, cudaGetErrorString(err));
            return false;
        }
    }
    LOG_DEBUG("Created %d BE streams", (int)be_streams_.size());
    
    return true;
}

void Scheduler::destroy_streams() {
    if (hp_stream_) {
        cudaStreamDestroy(hp_stream_);
        hp_stream_ = nullptr;
    }
    
    for (auto& stream : be_streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    be_streams_.clear();
}

void Scheduler::start() {
    if (!initialized_.load()) {
        LOG_ERROR("Scheduler not initialized");
        return;
    }
    
    if (running_.load()) {
        LOG_WARN("Scheduler already running");
        return;
    }
    
    running_.store(true);
    
    // 为每个客户端启动一个调度器线程
    for (int i = 0; i < num_clients_; i++) {
        threads_.emplace_back(&Scheduler::run_client, this, i);
        LOG_INFO("Started scheduler thread for client %d", i);
    }
}

void Scheduler::stop() {
    if (!running_.load()) return;
    
    LOG_INFO("Stopping scheduler...");
    running_.store(false);
    
    // 唤醒所有等待的线程
    g_capture_state.scheduler_cv.notify_all();
    be_schedule_cv_.notify_all();
}

void Scheduler::join() {
    for (auto& t : threads_) {
        if (t.joinable()) {
            t.join();
        }
    }
    threads_.clear();
    LOG_INFO("All scheduler threads joined");
}

// ============================================================================
// 每个客户端的调度器线程
// ============================================================================

void Scheduler::run_client(int client_idx) {
    LOG_INFO("Scheduler thread for client %d started", client_idx);
    
    // 确定使用哪个 stream
    cudaStream_t my_stream;
    bool is_hp = (client_idx == 0);
    
    if (is_hp) {
        my_stream = hp_stream_;
    } else {
        my_stream = be_streams_[client_idx - 1];
    }
    
    while (running_.load()) {
        // 从对应的客户端队列取操作
        if (!g_capture_state.client_queues[client_idx]) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        
        OperationPtr op = g_capture_state.client_queues[client_idx]->try_pop();
        
        if (op) {
            if (is_hp) {
                // HP 操作：直接执行
                LOG_DEBUG("HP thread: executing op %lu type %d", op->op_id, (int)op->type);
                
                hp_task_running_.store(true);
                current_hp_op_ = op;
                
                cudaError_t err = execute_operation(op, my_stream);
                op->mark_completed(err);
                
                hp_task_running_.store(false);
                current_hp_op_ = nullptr;
                
                // HP 完成后，唤醒可能等待的 BE 线程
                be_schedule_cv_.notify_all();
                
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.hp_ops_scheduled++;
                }
            } else {
                // BE 操作：可能需要等待调度许可
                LOG_DEBUG("BE thread %d: checking schedule for op %lu", client_idx, op->op_id);
                
                // 调度决策
                bool allowed = false;
                {
                    std::unique_lock<std::mutex> lock(be_schedule_mutex_);
                    
                    // 等待直到允许执行或停止
                    be_schedule_cv_.wait(lock, [this, &op, &allowed]() {
                        if (!running_.load()) return true;
                        
                        // 检查是否允许执行
                        allowed = schedule_be(current_hp_op_, op);
                        return allowed || !hp_task_running_.load();
                    });
                    
                    if (!running_.load()) break;
                    
                    // 如果 HP 没在运行，也允许执行
                    if (!hp_task_running_.load()) {
                        allowed = true;
                    }
                }
                
                if (allowed) {
                    LOG_DEBUG("BE thread %d: executing op %lu", client_idx, op->op_id);
                    
                    active_be_count_.fetch_add(1);
                    cudaError_t err = execute_operation(op, my_stream);
                    active_be_count_.fetch_sub(1);
                    
                    op->mark_completed(err);
                    
                    {
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        stats_.be_ops_scheduled++;
                    }
                } else {
                    // 不允许执行，重新入队（简化处理：直接执行）
                    LOG_DEBUG("BE thread %d: forced execute op %lu", client_idx, op->op_id);
                    cudaError_t err = execute_operation(op, my_stream);
                    op->mark_completed(err);
                    
                    {
                        std::lock_guard<std::mutex> lock(stats_mutex_);
                        stats_.be_ops_scheduled++;
                    }
                }
            }
        } else {
            // 没有操作，等待
            std::unique_lock<std::mutex> lock(g_capture_state.scheduler_mutex);
            g_capture_state.scheduler_cv.wait_for(lock, 
                std::chrono::microseconds(config_.poll_interval_us),
                [this, client_idx] {
                    if (!running_.load()) return true;
                    return g_capture_state.client_queues[client_idx] && 
                           !g_capture_state.client_queues[client_idx]->empty();
                });
        }
    }
    
    LOG_INFO("Scheduler thread for client %d ending", client_idx);
    
    // 处理剩余操作
    while (g_capture_state.client_queues[client_idx] && 
           !g_capture_state.client_queues[client_idx]->empty()) {
        auto op = g_capture_state.client_queues[client_idx]->try_pop();
        if (op) {
            cudaError_t err = execute_operation(op, my_stream);
            op->mark_completed(err);
        }
    }
    
    // 等待 stream 完成
    cudaStreamSynchronize(my_stream);
}

// 保留原有的 run() 函数作为备用（单线程模式）
void Scheduler::run() {
    // 多线程模式下不使用此函数
    LOG_WARN("Single-threaded run() called in multi-threaded scheduler");
}

bool Scheduler::schedule_be(const OperationPtr& hp_op, const OperationPtr& be_op) {
    // 如果没有 HP 任务在执行，允许 BE
    if (!hp_task_running_.load()) {
        return true;
    }
    
    // 如果不启用干扰感知，允许并发（依赖 stream 优先级）
    if (!config_.interference_aware) {
        return true;
    }
    
    // 检查 BE 操作的 SM 需求
    int sm_needed = be_op ? be_op->sm_needed : 0;
    if (sm_needed <= 0) {
        sm_needed = config_.num_sms / 4;  // 默认 25%
    }
    
    if (sm_needed >= config_.get_sm_threshold()) {
        LOG_TRACE("BE op rejected: SM needed %d >= threshold %d", 
                  sm_needed, config_.get_sm_threshold());
        return false;
    }
    
    // 检查 profile 类型是否互补
    ProfileType hp_type = hp_op ? hp_op->profile_type : ProfileType::UNKNOWN;
    ProfileType be_type = be_op ? be_op->profile_type : ProfileType::UNKNOWN;
    
    if (!is_complementary(hp_type, be_type)) {
        LOG_TRACE("BE op rejected: profiles not complementary");
        return false;
    }
    
    // 检查累计 BE 时间是否超过阈值
    float be_duration = be_op ? be_op->estimated_duration_ms : 0.1f;
    if (be_duration <= 0) {
        be_duration = 0.1f;
    }
    
    {
        std::lock_guard<std::mutex> lock(cumulative_mutex_);
        if (cumulative_be_duration_ms_ + be_duration > config_.get_dur_threshold_ms()) {
            LOG_TRACE("BE op rejected: cumulative duration %.3f + %.3f > threshold %.3f",
                      cumulative_be_duration_ms_, be_duration, config_.get_dur_threshold_ms());
            return false;
        }
        
        // 更新累计时间
        cumulative_be_duration_ms_ += be_duration;
    }
    
    return true;
}

bool Scheduler::is_complementary(ProfileType hp_type, ProfileType be_type) {
    // 如果任一类型为 UNKNOWN，认为可能互补
    if (hp_type == ProfileType::UNKNOWN || be_type == ProfileType::UNKNOWN) {
        return true;
    }
    
    // 一个 compute-bound，一个 memory-bound 时互补
    return (hp_type == ProfileType::COMPUTE_BOUND && be_type == ProfileType::MEMORY_BOUND) ||
           (hp_type == ProfileType::MEMORY_BOUND && be_type == ProfileType::COMPUTE_BOUND);
}

void Scheduler::process_hp_operation(OperationPtr op) {
    // 多线程模式下由 run_client() 处理
}

void Scheduler::process_be_operation(OperationPtr op) {
    // 多线程模式下由 run_client() 处理
}

void Scheduler::check_outstanding_kernels() {
    std::lock_guard<std::mutex> lock(outstanding_mutex_);
    
    auto it = outstanding_kernels_.begin();
    while (it != outstanding_kernels_.end()) {
        cudaError_t status = cudaEventQuery(it->end_event);
        
        if (status == cudaSuccess) {
            float elapsed_ms = 0;
            cudaEventElapsedTime(&elapsed_ms, it->start_event, it->end_event);
            
            // 从累计时间中减去
            float est = it->estimated_duration_ms > 0 ? it->estimated_duration_ms : 0.1f;
            {
                std::lock_guard<std::mutex> lock(cumulative_mutex_);
                cumulative_be_duration_ms_ = std::max(0.0f, cumulative_be_duration_ms_ - est);
            }
            
            if (it->op && !it->op->completed.load()) {
                it->op->mark_completed(it->op->result);
            }
            
            cudaEventDestroy(it->start_event);
            cudaEventDestroy(it->end_event);
            
            it = outstanding_kernels_.erase(it);
        } else if (status == cudaErrorNotReady) {
            ++it;
        } else {
            LOG_ERROR("Event query error: %s", cudaGetErrorString(status));
            if (it->op && !it->op->completed.load()) {
                it->op->mark_completed(status);
            }
            cudaEventDestroy(it->start_event);
            cudaEventDestroy(it->end_event);
            it = outstanding_kernels_.erase(it);
        }
    }
}

cudaError_t Scheduler::execute_operation(OperationPtr op, cudaStream_t stream) {
    op->started.store(true);
    
    LOG_DEBUG("Executing op %lu type %d on stream %p", op->op_id, (int)op->type, stream);
    
    switch (op->type) {
        case OperationType::KERNEL_LAUNCH:
        case OperationType::MALLOC:
        case OperationType::FREE:
        case OperationType::MEMCPY:
        case OperationType::MEMCPY_ASYNC:
        case OperationType::MEMSET:
        case OperationType::MEMSET_ASYNC:
        case OperationType::DEVICE_SYNC:
        case OperationType::STREAM_SYNC:
            return execute_cuda_operation(op, stream);
            
        case OperationType::CUDNN_CONV_FWD:
        case OperationType::CUDNN_CONV_BWD_DATA:
        case OperationType::CUDNN_CONV_BWD_FILTER:
        case OperationType::CUDNN_BATCHNORM_FWD:
        case OperationType::CUDNN_BATCHNORM_BWD:
            return execute_cudnn_operation(op, stream) == 0 ? cudaSuccess : cudaErrorUnknown;
            
        case OperationType::CUBLAS_SGEMM:
        case OperationType::CUBLAS_SGEMM_BATCHED:
        case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED:
            return execute_cublas_operation(op, stream) == 0 ? cudaSuccess : cudaErrorUnknown;
            
        default:
            LOG_ERROR("Unknown operation type: %d", (int)op->type);
            return cudaErrorUnknown;
    }
}

// ============================================================================
// 便捷函数
// ============================================================================

bool start_scheduler(int num_clients, const SchedulerConfig& config) {
    if (init_capture_layer(num_clients) != 0) {
        LOG_ERROR("Failed to initialize capture layer");
        return false;
    }
    
    if (!g_scheduler.init(num_clients, config)) {
        LOG_ERROR("Failed to initialize scheduler");
        return false;
    }
    
    g_scheduler.start();
    return true;
}

void stop_scheduler() {
    g_scheduler.stop();
    g_scheduler.join();
    shutdown_capture_layer();
}

} // namespace orion

// ============================================================================
// C 接口
// ============================================================================

extern "C" {

int orion_start_scheduler(int num_clients) {
    orion::SchedulerConfig config;
    return orion::start_scheduler(num_clients, config) ? 0 : -1;
}

void orion_stop_scheduler() {
    orion::stop_scheduler();
}

void orion_set_hp_latency(float latency_ms) {
    // 需要在启动前设置
}

} // extern "C"
