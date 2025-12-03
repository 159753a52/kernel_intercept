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
    
    // 预分配 outstanding kernels 空间
    outstanding_kernels_.reserve(64);
    
    initialized_.store(true);
    LOG_INFO("Scheduler initialized with %d clients", num_clients);
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
    thread_ = std::thread(&Scheduler::run, this);
    LOG_INFO("Scheduler thread started");
}

void Scheduler::stop() {
    if (!running_.load()) return;
    
    LOG_INFO("Stopping scheduler...");
    running_.store(false);
    
    // 唤醒调度器
    g_capture_state.scheduler_cv.notify_all();
}

void Scheduler::join() {
    if (thread_.joinable()) {
        thread_.join();
        LOG_INFO("Scheduler thread joined");
    }
}

void Scheduler::run() {
    LOG_INFO("Scheduler main loop started");
    
    while (running_.load()) {
        bool has_work = false;
        
        // 处理所有客户端的队列
        for (int i = 0; i < num_clients_; i++) {
            if (!g_capture_state.client_queues[i]) continue;
            
            OperationPtr op = g_capture_state.client_queues[i]->try_pop();
            if (op) {
                // 为不同客户端分配不同的 stream
                // client 0 = HP (高优先级), client 1+ = BE (低优先级)
                cudaStream_t stream = (i == 0) ? hp_stream_ : be_streams_[i - 1];
                
                LOG_DEBUG("Scheduler: client %d op %lu type %d on stream %p", 
                         i, op->op_id, (int)op->type, stream);
                
                cudaError_t err = execute_operation(op, stream);
                LOG_DEBUG("Scheduler: op %lu executed with result %d", op->op_id, (int)err);
                op->mark_completed(err);
                has_work = true;
                
                if (i == 0) {
                    stats_.hp_ops_scheduled++;
                } else {
                    stats_.be_ops_scheduled++;
                }
            }
        }
        
        // 如果没有工作，短暂等待
        if (!has_work) {
            std::unique_lock<std::mutex> lock(g_capture_state.scheduler_mutex);
            g_capture_state.scheduler_cv.wait_for(lock, 
                std::chrono::microseconds(config_.poll_interval_us),
                [this] {
                    if (!running_.load()) return true;
                    // 检查任何客户端队列是否有操作
                    for (int i = 0; i < num_clients_; i++) {
                        if (g_capture_state.client_queues[i] && 
                            !g_capture_state.client_queues[i]->empty()) {
                            return true;
                        }
                    }
                    return false;
                });
        }
    }
    
    LOG_INFO("Scheduler main loop ended");
    
    // 处理剩余操作
    LOG_INFO("Processing remaining operations...");
    for (int i = 0; i < num_clients_; i++) {
        while (g_capture_state.client_queues[i] && 
               !g_capture_state.client_queues[i]->empty()) {
            auto op = g_capture_state.client_queues[i]->try_pop();
            if (op) {
                cudaStream_t stream = (i == 0) ? hp_stream_ : be_streams_[i - 1];
                cudaError_t err = execute_operation(op, stream);
                op->mark_completed(err);
            }
        }
    }
    
    // 等待所有 stream 完成
    cudaStreamSynchronize(hp_stream_);
    for (auto& stream : be_streams_) {
        cudaStreamSynchronize(stream);
    }
}

bool Scheduler::schedule_be(const OperationPtr& hp_op, const OperationPtr& be_op) {
    // 如果没有 HP 任务在执行，允许 BE
    if (!hp_task_running_.load()) {
        return true;
    }
    
    // 如果不启用干扰感知，默认拒绝
    if (!config_.interference_aware) {
        return false;
    }
    
    // 检查 BE 操作的 SM 需求
    int sm_needed = be_op->sm_needed;
    if (sm_needed <= 0) {
        // 没有 profile 信息，使用默认值
        sm_needed = config_.num_sms / 4;  // 假设占用 25%
    }
    
    if (sm_needed >= config_.get_sm_threshold()) {
        LOG_TRACE("BE op rejected: SM needed %d >= threshold %d", 
                  sm_needed, config_.get_sm_threshold());
        return false;
    }
    
    // 检查 profile 类型是否互补
    ProfileType hp_type = hp_op ? hp_op->profile_type : ProfileType::UNKNOWN;
    ProfileType be_type = be_op->profile_type;
    
    if (!is_complementary(hp_type, be_type)) {
        LOG_TRACE("BE op rejected: profiles not complementary");
        return false;
    }
    
    // 检查累计 BE 时间是否超过阈值
    float be_duration = be_op->estimated_duration_ms;
    if (be_duration <= 0) {
        be_duration = 0.1f;  // 默认 0.1ms
    }
    
    if (cumulative_be_duration_ms_ + be_duration > config_.get_dur_threshold_ms()) {
        LOG_TRACE("BE op rejected: cumulative duration %.3f + %.3f > threshold %.3f",
                  cumulative_be_duration_ms_, be_duration, config_.get_dur_threshold_ms());
        return false;
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
    LOG_TRACE("Processing HP op %lu type %s", op->op_id, op_type_name(op->type));
    
    hp_task_running_.store(true);
    current_hp_op_ = op;
    
    // 执行操作
    cudaError_t err = execute_operation(op, hp_stream_);
    
    // 对于同步操作，等待 stream 完成
    if (op->type == OperationType::DEVICE_SYNC ||
        op->type == OperationType::STREAM_SYNC ||
        op->type == OperationType::MEMCPY ||
        op->type == OperationType::MALLOC ||
        op->type == OperationType::FREE) {
        cudaStreamSynchronize(hp_stream_);
    }
    
    hp_task_running_.store(false);
    current_hp_op_ = nullptr;
    
    // 重置 BE 累计时间
    cumulative_be_duration_ms_ = 0.0f;
    
    // 标记完成
    op->mark_completed(err);
    stats_.hp_ops_scheduled++;
    
    LOG_TRACE("HP op %lu completed with result %d", op->op_id, (int)err);
}

void Scheduler::process_be_operation(OperationPtr op) {
    LOG_TRACE("Processing BE op %lu type %s from client %d", 
              op->op_id, op_type_name(op->type), op->client_idx);
    
    int be_idx = op->client_idx - 1;
    if (be_idx < 0 || be_idx >= (int)be_streams_.size()) {
        LOG_ERROR("Invalid BE client index: %d", op->client_idx);
        op->mark_completed(cudaErrorInvalidValue);
        return;
    }
    
    cudaStream_t stream = be_streams_[be_idx];
    
    // 创建事件用于跟踪
    OutstandingKernel outstanding;
    outstanding.op = op;
    outstanding.estimated_duration_ms = op->estimated_duration_ms;
    outstanding.submit_time = std::chrono::steady_clock::now();
    
    cudaEventCreate(&outstanding.start_event);
    cudaEventCreate(&outstanding.end_event);
    
    // 记录开始事件
    cudaEventRecord(outstanding.start_event, stream);
    
    // 执行操作
    cudaError_t err = execute_operation(op, stream);
    
    // 记录结束事件
    cudaEventRecord(outstanding.end_event, stream);
    
    // 添加到 outstanding 列表
    outstanding_kernels_.push_back(outstanding);
    
    // 更新累计时间
    cumulative_be_duration_ms_ += op->estimated_duration_ms > 0 ? 
                                   op->estimated_duration_ms : 0.1f;
    
    // 对于同步操作，立即等待完成
    if (op->type == OperationType::DEVICE_SYNC ||
        op->type == OperationType::STREAM_SYNC ||
        op->type == OperationType::MEMCPY ||
        op->type == OperationType::MALLOC ||
        op->type == OperationType::FREE) {
        cudaStreamSynchronize(stream);
        op->mark_completed(err);
    } else {
        // 异步操作，稍后检查完成
        op->result = err;
    }
    
    stats_.be_ops_scheduled++;
}

void Scheduler::check_outstanding_kernels() {
    auto it = outstanding_kernels_.begin();
    while (it != outstanding_kernels_.end()) {
        cudaError_t status = cudaEventQuery(it->end_event);
        
        if (status == cudaSuccess) {
            // Kernel 已完成
            float elapsed_ms = 0;
            cudaEventElapsedTime(&elapsed_ms, it->start_event, it->end_event);
            
            // 从累计时间中减去
            cumulative_be_duration_ms_ -= it->estimated_duration_ms > 0 ? 
                                          it->estimated_duration_ms : 0.1f;
            cumulative_be_duration_ms_ = std::max(0.0f, cumulative_be_duration_ms_);
            
            // 标记完成
            if (it->op && !it->op->completed.load()) {
                it->op->mark_completed(it->op->result);
            }
            
            // 销毁事件
            cudaEventDestroy(it->start_event);
            cudaEventDestroy(it->end_event);
            
            it = outstanding_kernels_.erase(it);
        } else if (status == cudaErrorNotReady) {
            // 还在执行
            ++it;
        } else {
            // 错误
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
    
    // 根据操作类型调用相应的执行函数，传递调度器分配的 stream
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
    // 先初始化拦截层
    if (init_capture_layer(num_clients) != 0) {
        LOG_ERROR("Failed to initialize capture layer");
        return false;
    }
    
    // 初始化并启动调度器
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
    // 需要在启动前设置，这里简化处理
    // 实际应该通过配置接口
}

} // extern "C"
