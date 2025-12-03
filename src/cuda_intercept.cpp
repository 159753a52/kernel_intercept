#include "gpu_capture.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <nvToolsExt.h>

// NVTX 颜色定义
#define NVTX_COLOR_CLIENT   0xFF00FF00  // 绿色 - 客户端线程
#define NVTX_COLOR_SCHEDULER 0xFFFF0000 // 红色 - 调度器线程
#define NVTX_COLOR_WAIT     0xFF0000FF  // 蓝色 - 等待

// NVTX 辅助函数
static inline void nvtx_push(const char* name, uint32_t color) {
    nvtxEventAttributes_t attr = {0};
    attr.version = NVTX_VERSION;
    attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    attr.colorType = NVTX_COLOR_ARGB;
    attr.color = color;
    attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
    attr.message.ascii = name;
    nvtxRangePushEx(&attr);
}

static inline void nvtx_pop() {
    nvtxRangePop();
}

// 线程局部重入保护：当在执行操作时，避免递归拦截 (非 static 以便其他文件引用)
thread_local bool tl_in_scheduler_execution = false;

// ============================================================================
// Kernel 参数大小缓存
// 由于 CUDA Driver API 查询开销较大，我们缓存结果
// ============================================================================
static std::unordered_map<const void*, size_t> g_kernel_param_size_cache;
static std::mutex g_kernel_cache_mutex;

// 获取 kernel 参数总大小 (使用固定大小策略)
static size_t get_kernel_param_size(const void* func) {
    // 先查缓存
    {
        std::lock_guard<std::mutex> lock(g_kernel_cache_mutex);
        auto it = g_kernel_param_size_cache.find(func);
        if (it != g_kernel_param_size_cache.end()) {
            return it->second;
        }
    }
    
    // 对于 CUDA Runtime API 启动的 kernel，我们无法直接获取参数大小
    // 使用保守的固定大小策略：256 字节足以覆盖绝大多数 kernel
    size_t param_size = 256;
    
    // 缓存结果
    {
        std::lock_guard<std::mutex> lock(g_kernel_cache_mutex);
        g_kernel_param_size_cache[func] = param_size;
    }
    
    return param_size;
}

namespace orion {

// ============================================================================
// 真实 CUDA 函数指针类型定义
// ============================================================================

using cudaMalloc_t = cudaError_t (*)(void**, size_t);
using cudaFree_t = cudaError_t (*)(void*);
using cudaMemcpy_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind);
using cudaMemcpyAsync_t = cudaError_t (*)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
using cudaMemset_t = cudaError_t (*)(void*, int, size_t);
using cudaMemsetAsync_t = cudaError_t (*)(void*, int, size_t, cudaStream_t);
using cudaLaunchKernel_t = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
using cudaDeviceSynchronize_t = cudaError_t (*)();
using cudaStreamSynchronize_t = cudaError_t (*)(cudaStream_t);
using cudaEventSynchronize_t = cudaError_t (*)(cudaEvent_t);

// ============================================================================
// 真实函数指针存储
// ============================================================================

static struct {
    cudaMalloc_t cudaMalloc;
    cudaFree_t cudaFree;
    cudaMemcpy_t cudaMemcpy;
    cudaMemcpyAsync_t cudaMemcpyAsync;
    cudaMemset_t cudaMemset;
    cudaMemsetAsync_t cudaMemsetAsync;
    cudaLaunchKernel_t cudaLaunchKernel;
    cudaDeviceSynchronize_t cudaDeviceSynchronize;
    cudaStreamSynchronize_t cudaStreamSynchronize;
    cudaEventSynchronize_t cudaEventSynchronize;
    bool initialized;
    std::mutex init_mutex;
} g_real_funcs = {nullptr};

// CUDA runtime 库句柄
static void* g_cudart_handle = nullptr;

// 获取 CUDA 函数
static void* get_cuda_func(const char* name) {
    if (!g_cudart_handle) {
        const char* lib_paths[] = {
            "libcudart.so.12",
            "libcudart.so.11",
            "libcudart.so",
            nullptr
        };
        
        for (int i = 0; lib_paths[i]; i++) {
            g_cudart_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_NOLOAD);
            if (g_cudart_handle) break;
        }
        
        if (!g_cudart_handle) {
            for (int i = 0; lib_paths[i]; i++) {
                g_cudart_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_GLOBAL);
                if (g_cudart_handle) break;
            }
        }
    }
    
    if (g_cudart_handle) {
        void* fn = dlsym(g_cudart_handle, name);
        if (fn) return fn;
    }
    
    return dlsym(RTLD_DEFAULT, name);
}

// 初始化真实函数指针
static void init_real_functions() {
    std::lock_guard<std::mutex> lock(g_real_funcs.init_mutex);
    if (g_real_funcs.initialized) return;
    
    g_real_funcs.cudaMalloc = (cudaMalloc_t)get_cuda_func("cudaMalloc");
    g_real_funcs.cudaFree = (cudaFree_t)get_cuda_func("cudaFree");
    g_real_funcs.cudaMemcpy = (cudaMemcpy_t)get_cuda_func("cudaMemcpy");
    g_real_funcs.cudaMemcpyAsync = (cudaMemcpyAsync_t)get_cuda_func("cudaMemcpyAsync");
    g_real_funcs.cudaMemset = (cudaMemset_t)get_cuda_func("cudaMemset");
    g_real_funcs.cudaMemsetAsync = (cudaMemsetAsync_t)get_cuda_func("cudaMemsetAsync");
    g_real_funcs.cudaLaunchKernel = (cudaLaunchKernel_t)get_cuda_func("cudaLaunchKernel");
    g_real_funcs.cudaDeviceSynchronize = (cudaDeviceSynchronize_t)get_cuda_func("cudaDeviceSynchronize");
    g_real_funcs.cudaStreamSynchronize = (cudaStreamSynchronize_t)get_cuda_func("cudaStreamSynchronize");
    g_real_funcs.cudaEventSynchronize = (cudaEventSynchronize_t)get_cuda_func("cudaEventSynchronize");
    
    g_real_funcs.initialized = true;
    LOG_DEBUG("Real CUDA functions initialized");
}

// 获取真实函数 (带延迟初始化)
#define GET_REAL_FUNC(name) \
    do { \
        if (!g_real_funcs.initialized) init_real_functions(); \
        if (!g_real_funcs.name) { \
            LOG_ERROR("Failed to get real " #name); \
            return cudaErrorUnknown; \
        } \
    } while(0)

// 安全透传宏: 如果调度器未初始化或正在执行操作，直接调用真实函数
#define SAFE_PASSTHROUGH(func_name, ...) \
    do { \
        if (!g_capture_state.initialized.load() || tl_in_scheduler_execution) { \
            if (!g_real_funcs.initialized) init_real_functions(); \
            if (g_real_funcs.func_name) { \
                return g_real_funcs.func_name(__VA_ARGS__); \
            } \
            return cudaErrorUnknown; \
        } \
    } while(0)

// ============================================================================
// 执行真实操作的函数 (由调度器调用)
// ============================================================================

cudaError_t execute_malloc(OperationPtr op) {
    GET_REAL_FUNC(cudaMalloc);
    auto& p = std::get<MallocParams>(op->params);
    cudaError_t err = g_real_funcs.cudaMalloc(p.devPtr, p.size);
    op->result_ptr = *p.devPtr;
    return err;
}

cudaError_t execute_free(OperationPtr op) {
    GET_REAL_FUNC(cudaFree);
    auto& p = std::get<FreeParams>(op->params);
    return g_real_funcs.cudaFree(p.devPtr);
}

cudaError_t execute_memcpy(OperationPtr op) {
    auto& p = std::get<MemcpyParams>(op->params);
    if (p.is_async) {
        GET_REAL_FUNC(cudaMemcpyAsync);
        return g_real_funcs.cudaMemcpyAsync(p.dst, p.src, p.count, p.kind, p.stream);
    } else {
        GET_REAL_FUNC(cudaMemcpy);
        return g_real_funcs.cudaMemcpy(p.dst, p.src, p.count, p.kind);
    }
}

cudaError_t execute_memset(OperationPtr op) {
    auto& p = std::get<MemsetParams>(op->params);
    if (p.is_async) {
        GET_REAL_FUNC(cudaMemsetAsync);
        return g_real_funcs.cudaMemsetAsync(p.devPtr, p.value, p.count, p.stream);
    } else {
        GET_REAL_FUNC(cudaMemset);
        return g_real_funcs.cudaMemset(p.devPtr, p.value, p.count);
    }
}

// 用于在调度器线程中标记 kernel 执行的事件
static cudaEvent_t g_scheduler_event = nullptr;
static std::once_flag g_event_init_flag;

cudaError_t execute_kernel_launch(OperationPtr op, cudaStream_t scheduler_stream) {
    GET_REAL_FUNC(cudaLaunchKernel);
    
    // 初始化事件（只执行一次）
    std::call_once(g_event_init_flag, []() {
        cudaEventCreate(&g_scheduler_event);
    });
    
    // NVTX 标记调度器线程执行 kernel
    nvtxRangePush("Scheduler:execute_kernel");
    
    if (op->params.index() != 0) {
        LOG_ERROR("Wrong variant index! Expected 0 (KernelLaunchParams), got %zu", op->params.index());
        nvtxRangePop();
        return cudaErrorUnknown;
    }
    
    auto& p = std::get<KernelLaunchParams>(op->params);
    void** args = p.get_args();
    
    // 使用调度器分配的 stream 而不是客户端的 stream
    cudaStream_t stream_to_use = scheduler_stream ? scheduler_stream : p.stream;
    
    cudaError_t result = g_real_funcs.cudaLaunchKernel(
        p.func, p.gridDim, p.blockDim,
        args,
        p.sharedMem, stream_to_use
    );
    
    nvtxRangePop();
    return result;
}

cudaError_t execute_device_sync(OperationPtr op) {
    GET_REAL_FUNC(cudaDeviceSynchronize);
    (void)op;
    return g_real_funcs.cudaDeviceSynchronize();
}

cudaError_t execute_stream_sync(OperationPtr op) {
    GET_REAL_FUNC(cudaStreamSynchronize);
    auto& p = std::get<SyncParams>(op->params);
    return g_real_funcs.cudaStreamSynchronize(p.stream);
}

// 导出给调度器使用
cudaError_t execute_cuda_operation(OperationPtr op, cudaStream_t scheduler_stream) {
    LOG_DEBUG("execute_cuda_operation: op type=%d, stream=%p", (int)op->type, scheduler_stream);
    
    // 设置重入标志，防止执行过程中的 CUDA 调用被再次拦截
    tl_in_scheduler_execution = true;
    
    cudaError_t result;
    switch (op->type) {
        case OperationType::MALLOC:
            result = execute_malloc(op);
            break;
        case OperationType::FREE:
            result = execute_free(op);
            break;
        case OperationType::MEMCPY:
        case OperationType::MEMCPY_ASYNC:
            result = execute_memcpy(op);
            break;
        case OperationType::MEMSET:
        case OperationType::MEMSET_ASYNC:
            result = execute_memset(op);
            break;
        case OperationType::KERNEL_LAUNCH:
            LOG_DEBUG("Entering execute_kernel_launch with stream=%p", scheduler_stream);
            result = execute_kernel_launch(op, scheduler_stream);
            LOG_DEBUG("execute_kernel_launch returned %d", (int)result);
            break;
        case OperationType::DEVICE_SYNC:
            result = execute_device_sync(op);
            break;
        case OperationType::STREAM_SYNC:
            result = execute_stream_sync(op);
            break;
        default:
            LOG_ERROR("Unknown operation type for execution: %d", (int)op->type);
            result = cudaErrorUnknown;
            break;
    }
    
    tl_in_scheduler_execution = false;
    return result;
}

// 直接调用真实函数的版本 (用于非管理线程)
cudaError_t real_cudaMalloc(void** devPtr, size_t size) {
    GET_REAL_FUNC(cudaMalloc);
    return g_real_funcs.cudaMalloc(devPtr, size);
}

cudaError_t real_cudaFree(void* devPtr) {
    GET_REAL_FUNC(cudaFree);
    return g_real_funcs.cudaFree(devPtr);
}

cudaError_t real_cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    GET_REAL_FUNC(cudaMemcpy);
    return g_real_funcs.cudaMemcpy(dst, src, count, kind);
}

cudaError_t real_cudaMemcpyAsync(void* dst, const void* src, size_t count, 
                                  cudaMemcpyKind kind, cudaStream_t stream) {
    GET_REAL_FUNC(cudaMemcpyAsync);
    return g_real_funcs.cudaMemcpyAsync(dst, src, count, kind, stream);
}

cudaError_t real_cudaMemset(void* devPtr, int value, size_t count) {
    GET_REAL_FUNC(cudaMemset);
    return g_real_funcs.cudaMemset(devPtr, value, count);
}

cudaError_t real_cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    GET_REAL_FUNC(cudaMemsetAsync);
    return g_real_funcs.cudaMemsetAsync(devPtr, value, count, stream);
}

cudaError_t real_cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                                   void** args, size_t sharedMem, cudaStream_t stream) {
    GET_REAL_FUNC(cudaLaunchKernel);
    return g_real_funcs.cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t real_cudaDeviceSynchronize() {
    GET_REAL_FUNC(cudaDeviceSynchronize);
    return g_real_funcs.cudaDeviceSynchronize();
}

cudaError_t real_cudaStreamSynchronize(cudaStream_t stream) {
    GET_REAL_FUNC(cudaStreamSynchronize);
    return g_real_funcs.cudaStreamSynchronize(stream);
}

} // namespace orion

// ============================================================================
// CUDA API Wrappers (LD_PRELOAD 拦截点)
// ============================================================================

extern "C" {

/**
 * cudaMalloc wrapper
 * 
 * 对于被管理的线程:
 * 1. 创建 OperationRecord
 * 2. 提交到队列
 * 3. 等待调度器执行完成
 * 4. 返回结果
 * 
 * 对于非管理线程:
 * 直接调用真实的 cudaMalloc
 */
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaMalloc, devPtr, size);
    
    if (!is_capture_enabled()) {
        return real_cudaMalloc(devPtr, size);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return real_cudaMalloc(devPtr, size);
    }
    
    auto op = create_operation(client_idx, OperationType::MALLOC);
    if (!op) return real_cudaMalloc(devPtr, size);
    
    op->params = MallocParams{devPtr, size};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * cudaFree wrapper
 */
cudaError_t cudaFree(void* devPtr) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaFree, devPtr);
    
    if (!is_capture_enabled()) return real_cudaFree(devPtr);
    int client_idx = get_current_client_idx();
    if (client_idx < 0) return real_cudaFree(devPtr);
    
    auto op = create_operation(client_idx, OperationType::FREE);
    if (!op) return real_cudaFree(devPtr);
    
    op->params = FreeParams{devPtr};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * cudaMemcpy wrapper
 * 
 * cudaMemcpy 有隐式同步语义 (对于 D2H 和 H2D)
 * 通过队列调度来维持这个语义
 */
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaMemcpy, dst, src, count, kind);
    
    if (!is_capture_enabled()) return real_cudaMemcpy(dst, src, count, kind);
    int client_idx = get_current_client_idx();
    if (client_idx < 0) return real_cudaMemcpy(dst, src, count, kind);
    
    auto op = create_operation(client_idx, OperationType::MEMCPY);
    if (!op) return real_cudaMemcpy(dst, src, count, kind);
    
    op->params = MemcpyParams{dst, src, count, kind, nullptr, false};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * cudaMemcpyAsync wrapper
 */
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, 
                            cudaMemcpyKind kind, cudaStream_t stream) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaMemcpyAsync, dst, src, count, kind, stream);
    
    if (!is_capture_enabled()) return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    int client_idx = get_current_client_idx();
    if (client_idx < 0) return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    
    auto op = create_operation(client_idx, OperationType::MEMCPY_ASYNC);
    if (!op) return real_cudaMemcpyAsync(dst, src, count, kind, stream);
    
    op->params = MemcpyParams{dst, src, count, kind, stream, true};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * cudaMemset wrapper
 */
cudaError_t cudaMemset(void* devPtr, int value, size_t count) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaMemset, devPtr, value, count);
    
    if (!is_capture_enabled()) return real_cudaMemset(devPtr, value, count);
    int client_idx = get_current_client_idx();
    if (client_idx < 0) return real_cudaMemset(devPtr, value, count);
    
    auto op = create_operation(client_idx, OperationType::MEMSET);
    if (!op) return real_cudaMemset(devPtr, value, count);
    
    op->params = MemsetParams{devPtr, value, count, nullptr, false};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * cudaMemsetAsync wrapper
 */
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaMemsetAsync, devPtr, value, count, stream);
    
    if (!is_capture_enabled()) return real_cudaMemsetAsync(devPtr, value, count, stream);
    int client_idx = get_current_client_idx();
    if (client_idx < 0) return real_cudaMemsetAsync(devPtr, value, count, stream);
    
    auto op = create_operation(client_idx, OperationType::MEMSET_ASYNC);
    if (!op) return real_cudaMemsetAsync(devPtr, value, count, stream);
    
    op->params = MemsetParams{devPtr, value, count, stream, true};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * cudaLaunchKernel wrapper
 * 
 * 这是最关键的拦截点，所有 CUDA kernel 最终都通过这里发起
 * 
 * 实现调度器执行的关键：
 * 1. 深拷贝所有 kernel 参数
 * 2. 提交到调度器队列
 * 3. 客户端线程等待
 * 4. 调度器线程执行真实的 cudaLaunchKernel
 */
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim,
                             void** args, size_t sharedMem, cudaStream_t stream) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaLaunchKernel, func, gridDim, blockDim, args, sharedMem, stream);
    
    if (!is_capture_enabled()) return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    int client_idx = get_current_client_idx();
    if (client_idx < 0) return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    
    // NVTX 标记客户端线程提交 kernel
    nvtxRangePush("Client:submit_kernel");
    
    // 使用新接口避免竞态条件
    auto op = create_operation(client_idx, OperationType::KERNEL_LAUNCH);
    if (!op) {
        nvtxRangePop();
        return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }
    
    // 先设置 params
    KernelLaunchParams kp;
    kp.func = func;
    kp.gridDim = gridDim;
    kp.blockDim = blockDim;
    kp.sharedMem = sharedMem;
    kp.stream = stream;
    kp.original_args = args;
    kp.use_deep_copy = false;
    op->params = std::move(kp);
    
    enqueue_operation(op);
    
    // NVTX 标记等待调度器执行
    nvtxRangePush("Client:wait_scheduler");
    wait_operation(op);
    nvtxRangePop();  // 结束等待
    
    nvtxRangePop();  // 结束提交
    return op->result;
}

/**
 * cudaDeviceSynchronize wrapper
 * 
 * 显式同步操作，需要等待所有之前的操作完成
 */
cudaError_t cudaDeviceSynchronize(void) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaDeviceSynchronize);
    
    if (!is_capture_enabled()) return real_cudaDeviceSynchronize();
    int client_idx = get_current_client_idx();
    if (client_idx < 0) return real_cudaDeviceSynchronize();
    
    auto op = create_operation(client_idx, OperationType::DEVICE_SYNC);
    if (!op) return real_cudaDeviceSynchronize();
    
    op->params = SyncParams{nullptr, nullptr};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

/**
 * cudaStreamSynchronize wrapper
 */
cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    using namespace orion;
    
    SAFE_PASSTHROUGH(cudaStreamSynchronize, stream);
    
    if (!is_capture_enabled()) return real_cudaStreamSynchronize(stream);
    int client_idx = get_current_client_idx();
    if (client_idx < 0) return real_cudaStreamSynchronize(stream);
    
    auto op = create_operation(client_idx, OperationType::STREAM_SYNC);
    if (!op) return real_cudaStreamSynchronize(stream);
    
    op->params = SyncParams{stream, nullptr};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

} // extern "C"
