#ifndef ORION_GPU_CAPTURE_H
#define ORION_GPU_CAPTURE_H

#include "common.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <vector>
#include <functional>
#include <memory>
#include <variant>

namespace orion {

// ============================================================================
// 操作参数结构体
// ============================================================================

// Kernel 参数的最大总大小 (覆盖绝大多数情况)
constexpr size_t MAX_KERNEL_ARGS_SIZE = 4096;
// Kernel 最大参数个数
constexpr size_t MAX_KERNEL_ARGS_COUNT = 64;

struct KernelLaunchParams {
    const void* func;
    dim3 gridDim;
    dim3 blockDim;
    size_t sharedMem;
    cudaStream_t stream;
    
    // 方案 A: 保存原始 args 指针 (仅在客户端等待期间有效)
    void** original_args;
    
    // 方案 B: 深拷贝参数 (更安全，但需要知道参数布局)
    std::vector<uint8_t> args_buffer;
    std::vector<size_t> args_offsets;
    std::vector<size_t> args_sizes;
    size_t num_args;
    std::vector<void*> args_ptrs;
    bool use_deep_copy;
    
    KernelLaunchParams() 
        : func(nullptr), sharedMem(0), stream(nullptr)
        , original_args(nullptr), num_args(0), use_deep_copy(false) {
        gridDim = {1, 1, 1};
        blockDim = {1, 1, 1};
    }
    
    // 获取用于执行的 args 指针
    void** get_args() {
        if (use_deep_copy && num_args > 0) {
            // 重建 args 指针数组
            args_ptrs.resize(num_args);
            for (size_t i = 0; i < num_args; i++) {
                args_ptrs[i] = args_buffer.data() + args_offsets[i];
            }
            return args_ptrs.data();
        }
        return original_args;
    }
};

struct MallocParams {
    void** devPtr;       // 输出: 分配的设备指针地址
    size_t size;
};

struct FreeParams {
    void* devPtr;
};

struct MemcpyParams {
    void* dst;
    const void* src;
    size_t count;
    cudaMemcpyKind kind;
    cudaStream_t stream;  // 仅用于 async 版本
    bool is_async;
};

struct MemsetParams {
    void* devPtr;
    int value;
    size_t count;
    cudaStream_t stream;  // 仅用于 async 版本
    bool is_async;
};

struct SyncParams {
    cudaStream_t stream;  // 仅用于 stream sync
    cudaEvent_t event;    // 仅用于 event sync
};

// cuDNN 卷积参数 (简化版)
struct CudnnConvParams {
    void* handle;
    void* xDesc;
    const void* x;
    void* wDesc;
    const void* w;
    void* convDesc;
    int algo;
    void* workSpace;
    size_t workSpaceSizeInBytes;
    const void* alpha;
    const void* beta;
    void* yDesc;
    void* y;
};

// cuDNN BatchNorm 参数 (简化版)
struct CudnnBatchNormParams {
    void* handle;
    int mode;
    const void* alpha;
    const void* beta;
    void* xDesc;
    const void* x;
    void* yDesc;
    void* y;
    void* bnScaleBiasMeanVarDesc;
    const void* bnScale;
    const void* bnBias;
    double exponentialAverageFactor;
    void* resultRunningMean;
    void* resultRunningVariance;
    double epsilon;
    void* resultSaveMean;
    void* resultSaveInvVariance;
};

// cuBLAS GEMM 参数
struct CublasGemmParams {
    void* handle;
    int transa;
    int transb;
    int m, n, k;
    const void* alpha;
    const void* A;
    int lda;
    const void* B;
    int ldb;
    const void* beta;
    void* C;
    int ldc;
    // Batched GEMM 额外参数
    int batchCount;
    long long strideA, strideB, strideC;
    bool is_batched;
    bool is_strided;
};

// ============================================================================
// OperationRecord: 统一的操作记录
// ============================================================================

struct OperationRecord {
    OperationType type;
    uint64_t op_id;
    int client_idx;
    
    // 操作参数 (使用 variant)
    std::variant<
        KernelLaunchParams,
        MallocParams,
        FreeParams,
        MemcpyParams,
        MemsetParams,
        SyncParams,
        CudnnConvParams,
        CudnnBatchNormParams,
        CublasGemmParams
    > params;
    
    // Profiling 信息 (由 profiling 阶段填充)
    std::string kernel_id;
    float estimated_duration_ms;
    int sm_needed;
    ProfileType profile_type;
    
    // 执行状态和结果
    std::atomic<bool> completed{false};
    std::atomic<bool> started{false};
    cudaError_t result;
    void* result_ptr;  // malloc 的返回指针
    
    // 条件变量用于等待完成
    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    
    OperationRecord() : type(OperationType::UNKNOWN), op_id(0), client_idx(-1),
                        params(MallocParams{}),
                        estimated_duration_ms(0.0f), sm_needed(0),
                        profile_type(ProfileType::UNKNOWN),
                        result(cudaSuccess), result_ptr(nullptr) {
    }
    
    ~OperationRecord() = default;
    
    // 禁止拷贝
    OperationRecord(const OperationRecord&) = delete;
    OperationRecord& operator=(const OperationRecord&) = delete;
    
    // 禁止移动 (因为含有 mutex)
    OperationRecord(OperationRecord&&) = delete;
    OperationRecord& operator=(OperationRecord&&) = delete;
    
    // 等待操作完成
    void wait_completion() {
        std::unique_lock<std::mutex> lock(completion_mutex);
        completion_cv.wait(lock, [this] { return completed.load(); });
    }
    
    // 标记操作完成
    void mark_completed(cudaError_t res) {
        result = res;
        completed.store(true);
        completion_cv.notify_all();
    }
};

using OperationPtr = std::shared_ptr<OperationRecord>;

// ============================================================================
// Per-Client 队列
// ============================================================================

class ClientQueue {
public:
    ClientQueue() : shutdown_(false) {}
    
    // 提交操作到队列
    void push(OperationPtr op) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(op));
        }
        cv_.notify_one();
    }
    
    // 尝试取出操作 (非阻塞)
    OperationPtr try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return nullptr;
        OperationPtr op = std::move(queue_.front());
        queue_.pop();
        return op;
    }
    
    // 阻塞等待操作
    OperationPtr wait_pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !queue_.empty() || shutdown_; });
        if (shutdown_ && queue_.empty()) return nullptr;
        OperationPtr op = std::move(queue_.front());
        queue_.pop();
        return op;
    }
    
    // 查看队首操作 (不移除)
    OperationPtr peek() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) return nullptr;
        return queue_.front();
    }
    
    bool empty() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    void shutdown() {
        shutdown_ = true;
        cv_.notify_all();
    }
    
private:
    std::queue<OperationPtr> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> shutdown_;
};

// ============================================================================
// 全局拦截层状态
// ============================================================================

struct CaptureLayerState {
    // 是否已初始化
    std::atomic<bool> initialized{false};
    
    // 是否启用拦截
    std::atomic<bool> enabled{false};
    
    // Client 数量
    int num_clients{0};
    
    // Per-client 队列
    std::vector<std::unique_ptr<ClientQueue>> client_queues;
    
    // 操作 ID 生成器
    std::atomic<uint64_t> next_op_id{0};
    
    // 用于 block/unblock 的同步 (使用指针数组而非 vector)
    std::atomic<bool>* client_blocked{nullptr};
    std::mutex* client_mutexes{nullptr};
    std::condition_variable* client_cvs{nullptr};
    
    // 调度器通知 (有新操作时通知调度器)
    std::mutex scheduler_mutex;
    std::condition_variable scheduler_cv;
    
    // 关闭标志
    std::atomic<bool> shutdown{false};
    
    ~CaptureLayerState() {
        delete[] client_blocked;
        delete[] client_mutexes;
        delete[] client_cvs;
    }
};

// 全局状态
extern CaptureLayerState g_capture_state;

// ============================================================================
// API 函数声明
// ============================================================================

// 初始化拦截层
// num_clients: client 数量 (包括 HP 和 BE)
// 返回: 成功返回 0，失败返回 -1
int init_capture_layer(int num_clients);

// 关闭拦截层
void shutdown_capture_layer();

// 获取当前线程的 client index
// 返回: client index，如果未注册返回 -1
int get_current_client_idx();

// 设置当前线程的 client index
void set_current_client_idx(int idx);

// 创建操作 (不加入队列)
OperationPtr create_operation(int client_idx, OperationType type);

// 将操作加入队列
void enqueue_operation(OperationPtr op);

// 提交操作到队列 (兼容旧接口，创建+入队)
// 注意：此接口有竞态条件风险，推荐使用 create_operation + 设置 params + enqueue_operation
OperationPtr submit_operation(int client_idx, OperationType type);

// 等待操作完成
void wait_operation(OperationPtr op);

// 检查当前线程是否被拦截层管理
bool is_managed_thread();

// 检查拦截是否启用
bool is_capture_enabled();

// 启用/禁用拦截
void set_capture_enabled(bool enabled);

// block 接口 (给外部调用，如 Python 绑定)
// phase 参数用于区分不同的同步点
extern "C" void block(int phase);

// unblock 接口 (由调度器调用)
void unblock_client(int client_idx);

// 通知调度器有新操作
void notify_scheduler();

} // namespace orion

#endif // ORION_GPU_CAPTURE_H
