#ifndef ORION_SCHEDULER_H
#define ORION_SCHEDULER_H

#include "gpu_capture.h"
#include "kernel_profile.h"
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

namespace orion {

// ============================================================================
// 调度配置
// ============================================================================

struct SchedulerConfig {
    // SM 阈值: BE kernel 的最大 SM 占用比例
    float sm_threshold_ratio = 0.5f;
    
    // 时间阈值: BE kernel 累计执行时间占 HP 请求延迟的最大比例
    float dur_threshold_ratio = 0.025f;  // 2.5%
    
    // HP 请求的平均延迟 (ms)
    float hp_request_latency_ms = 10.0f;
    
    // 轮询间隔 (us)
    int poll_interval_us = 10;
    
    // 是否启用干扰感知调度
    bool interference_aware = true;
    
    // GPU 设备 ID
    int device_id = 0;
    
    // GPU SM 数量 (运行时获取)
    int num_sms = 0;
    
    // 计算实际阈值
    int get_sm_threshold() const {
        return static_cast<int>(num_sms * sm_threshold_ratio);
    }
    
    float get_dur_threshold_ms() const {
        return hp_request_latency_ms * dur_threshold_ratio;
    }
};

// ============================================================================
// Outstanding kernel 记录 (用于跟踪 BE kernel 完成状态)
// ============================================================================

struct OutstandingKernel {
    OperationPtr op;
    cudaEvent_t start_event;
    cudaEvent_t end_event;
    float estimated_duration_ms;
    std::chrono::steady_clock::time_point submit_time;
    
    OutstandingKernel() : op(nullptr), start_event(nullptr), end_event(nullptr),
                          estimated_duration_ms(0.0f) {}
};

// ============================================================================
// 调度器类
// ============================================================================

class Scheduler {
public:
    Scheduler();
    ~Scheduler();
    
    // 初始化调度器
    bool init(int num_clients, const SchedulerConfig& config = SchedulerConfig());
    
    // 启动调度器线程
    void start();
    
    // 停止调度器
    void stop();
    
    // 等待调度器线程结束
    void join();
    
    // 获取配置
    const SchedulerConfig& get_config() const { return config_; }
    
    // 设置 profile 表
    void set_profile_table(KernelProfileTable* table) { profile_table_ = table; }
    
    // 统计信息
    struct Stats {
        uint64_t hp_ops_scheduled = 0;
        uint64_t be_ops_scheduled = 0;
        uint64_t be_ops_rejected = 0;
        uint64_t total_wait_time_us = 0;
    };
    
    Stats get_stats() const { return stats_; }
    
private:
    // 主调度循环
    void run();
    
    // 执行操作
    cudaError_t execute_operation(OperationPtr op, cudaStream_t stream);
    
    // 调度决策: 是否允许 BE 操作并发执行
    bool schedule_be(const OperationPtr& hp_op, const OperationPtr& be_op);
    
    // 检查 profile 类型是否互补
    bool is_complementary(ProfileType hp_type, ProfileType be_type);
    
    // 处理 HP 操作
    void process_hp_operation(OperationPtr op);
    
    // 处理 BE 操作
    void process_be_operation(OperationPtr op);
    
    // 检查并回收已完成的 BE kernel
    void check_outstanding_kernels();
    
    // 创建 CUDA streams
    bool create_streams();
    
    // 销毁 CUDA streams
    void destroy_streams();
    
private:
    SchedulerConfig config_;
    
    // 调度器线程
    std::thread thread_;
    std::atomic<bool> running_;
    std::atomic<bool> initialized_;
    
    // CUDA streams
    cudaStream_t hp_stream_;                    // 高优先级 stream
    std::vector<cudaStream_t> be_streams_;      // Best-effort streams (每个 BE client 一个)
    
    // 当前状态
    std::atomic<bool> hp_task_running_;         // 是否有 HP 任务在执行
    OperationPtr current_hp_op_;                // 当前 HP 操作
    
    // Outstanding BE kernels
    std::vector<OutstandingKernel> outstanding_kernels_;
    float cumulative_be_duration_ms_;           // 累计 BE kernel 时间
    
    // Profile 表
    KernelProfileTable* profile_table_;
    
    // 统计
    Stats stats_;
    
    // Client 数量
    int num_clients_;
};

// ============================================================================
// 全局调度器实例
// ============================================================================

extern Scheduler g_scheduler;

// 便捷函数
bool start_scheduler(int num_clients, const SchedulerConfig& config = SchedulerConfig());
void stop_scheduler();

} // namespace orion

#endif // ORION_SCHEDULER_H
