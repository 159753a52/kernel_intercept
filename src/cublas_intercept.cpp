#include "gpu_capture.h"
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <cstdio>
#include <mutex>

// 从 cuda_intercept.cpp 导入的重入保护标志
extern thread_local bool tl_in_scheduler_execution;

// cuBLAS 类型定义
typedef void* cublasHandle_t;
typedef int cublasStatus_t;
typedef int cublasOperation_t;

#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1

namespace orion {

// ============================================================================
// cuBLAS 真实函数指针类型
// ============================================================================

using cublasSgemm_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*, const float*, int,
    const float*, int,
    const float*, float*, int);

using cublasSgemmBatched_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*, const float* const*, int,
    const float* const*, int,
    const float*, float* const*, int,
    int);

using cublasSgemmStridedBatched_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*, const float*, int, long long,
    const float*, int, long long,
    const float*, float*, int, long long,
    int);

using cublasHgemm_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, int,
    const void*, int,
    const void*, void*, int);

using cublasGemmEx_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, int, int,
    const void*, int, int,
    const void*, void*, int, int,
    int, int);

// cublasGemmStridedBatchedEx - 混合精度批量 GEMM
using cublasGemmStridedBatchedEx_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*,
    const void*, int, int, long long,
    const void*, int, int, long long,
    const void*,
    void*, int, int, long long,
    int,
    int, int);

// cublasHgemmStridedBatched - FP16 批量 GEMM
using cublasHgemmStridedBatched_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const void*, const void*, int, long long,
    const void*, int, long long,
    const void*, void*, int, long long,
    int);

// cublasSgemmEx - 扩展 FP32 GEMM
using cublasSgemmEx_t = cublasStatus_t (*)(
    cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int,
    const float*,
    const void*, int, int,
    const void*, int, int,
    const float*,
    void*, int, int);

// cublasSetStream - 设置 cuBLAS handle 的 stream
using cublasSetStream_t = cublasStatus_t (*)(cublasHandle_t, cudaStream_t);
using cublasGetStream_t = cublasStatus_t (*)(cublasHandle_t, cudaStream_t*);

// ============================================================================
// 真实函数指针存储
// ============================================================================

static struct {
    cublasSgemm_t cublasSgemm_v2;
    cublasSgemmBatched_t cublasSgemmBatched;
    cublasSgemmStridedBatched_t cublasSgemmStridedBatched;
    cublasHgemm_t cublasHgemm;
    cublasGemmEx_t cublasGemmEx;
    cublasGemmStridedBatchedEx_t cublasGemmStridedBatchedEx;
    cublasHgemmStridedBatched_t cublasHgemmStridedBatched;
    cublasSgemmEx_t cublasSgemmEx;
    cublasSetStream_t cublasSetStream_v2;
    cublasGetStream_t cublasGetStream_v2;
    bool initialized;
    std::mutex init_mutex;
} g_cublas_funcs = {nullptr};

static void* g_cublas_handle = nullptr;

static void* get_cublas_func(const char* name) {
    // 如果还没有打开 libcublas，尝试打开它
    if (!g_cublas_handle) {
        // 尝试多个可能的库路径
        const char* lib_paths[] = {
            "libcublas.so.12",
            "libcublas.so.11", 
            "libcublas.so",
            nullptr
        };
        
        for (int i = 0; lib_paths[i]; i++) {
            // 使用 RTLD_NOLOAD 获取已加载的库句柄
            g_cublas_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_NOLOAD);
            if (g_cublas_handle) {
                LOG_DEBUG("Found cuBLAS library: %s", lib_paths[i]);
                break;
            }
        }
        
        // 如果 RTLD_NOLOAD 失败，尝试正常加载
        if (!g_cublas_handle) {
            for (int i = 0; lib_paths[i]; i++) {
                g_cublas_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_GLOBAL);
                if (g_cublas_handle) {
                    LOG_DEBUG("Loaded cuBLAS library: %s", lib_paths[i]);
                    break;
                }
            }
        }
    }
    
    if (g_cublas_handle) {
        void* fn = dlsym(g_cublas_handle, name);
        if (fn) return fn;
    }
    
    // 备选：使用 RTLD_DEFAULT
    return dlsym(RTLD_DEFAULT, name);
}

static void init_cublas_functions() {
    std::lock_guard<std::mutex> lock(g_cublas_funcs.init_mutex);
    if (g_cublas_funcs.initialized) return;
    
    g_cublas_funcs.cublasSgemm_v2 = 
        (cublasSgemm_t)get_cublas_func("cublasSgemm_v2");
    g_cublas_funcs.cublasSgemmBatched = 
        (cublasSgemmBatched_t)get_cublas_func("cublasSgemmBatched");
    g_cublas_funcs.cublasSgemmStridedBatched = 
        (cublasSgemmStridedBatched_t)get_cublas_func("cublasSgemmStridedBatched");
    g_cublas_funcs.cublasHgemm = 
        (cublasHgemm_t)get_cublas_func("cublasHgemm");
    g_cublas_funcs.cublasGemmEx = 
        (cublasGemmEx_t)get_cublas_func("cublasGemmEx");
    g_cublas_funcs.cublasGemmStridedBatchedEx = 
        (cublasGemmStridedBatchedEx_t)get_cublas_func("cublasGemmStridedBatchedEx");
    g_cublas_funcs.cublasHgemmStridedBatched = 
        (cublasHgemmStridedBatched_t)get_cublas_func("cublasHgemmStridedBatched");
    g_cublas_funcs.cublasSgemmEx = 
        (cublasSgemmEx_t)get_cublas_func("cublasSgemmEx");
    g_cublas_funcs.cublasSetStream_v2 = 
        (cublasSetStream_t)get_cublas_func("cublasSetStream_v2");
    g_cublas_funcs.cublasGetStream_v2 = 
        (cublasGetStream_t)get_cublas_func("cublasGetStream_v2");
    
    g_cublas_funcs.initialized = true;
    if (g_cublas_funcs.cublasSgemm_v2) {
        LOG_DEBUG("cuBLAS functions initialized");
    } else {
        LOG_WARN("cuBLAS functions not found - interception disabled");
    }
}

#define GET_CUBLAS_FUNC(name) \
    do { \
        if (!g_cublas_funcs.initialized) init_cublas_functions(); \
        if (!g_cublas_funcs.name) { \
            LOG_ERROR("Failed to get real " #name); \
            return 1; \
        } \
    } while(0)

// ============================================================================
// 执行真实 cuBLAS 操作
// ============================================================================

cublasStatus_t execute_cublas_sgemm(OperationPtr op) {
    GET_CUBLAS_FUNC(cublasSgemm_v2);
    auto& p = std::get<CublasGemmParams>(op->params);
    return g_cublas_funcs.cublasSgemm_v2(
        (cublasHandle_t)p.handle,
        (cublasOperation_t)p.transa, (cublasOperation_t)p.transb,
        p.m, p.n, p.k,
        (const float*)p.alpha, (const float*)p.A, p.lda,
        (const float*)p.B, p.ldb,
        (const float*)p.beta, (float*)p.C, p.ldc
    );
}

cublasStatus_t execute_cublas_operation(OperationPtr op, cudaStream_t scheduler_stream) {
    // 确保 cuBLAS 函数已初始化
    if (!g_cublas_funcs.initialized) {
        init_cublas_functions();
    }
    
    // 设置重入标志，防止 cuBLAS 内部的 cudaLaunchKernel 被再次拦截
    tl_in_scheduler_execution = true;
    
    cublasStatus_t result = CUBLAS_STATUS_SUCCESS;
    
    // 获取 handle
    cublasHandle_t handle = nullptr;
    cudaStream_t original_stream = nullptr;
    
    switch (op->type) {
        case OperationType::CUBLAS_SGEMM: {
            auto& p = std::get<CublasGemmParams>(op->params);
            handle = (cublasHandle_t)p.handle;
            break;
        }
        case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED: {
            auto& p = std::get<CublasGemmParams>(op->params);
            handle = (cublasHandle_t)p.handle;
            break;
        }
        default:
            break;
    }
    
    // 保存原始 stream 并设置调度器的 stream
    if (handle && scheduler_stream && g_cublas_funcs.cublasSetStream_v2 && g_cublas_funcs.cublasGetStream_v2) {
        g_cublas_funcs.cublasGetStream_v2(handle, &original_stream);
        g_cublas_funcs.cublasSetStream_v2(handle, scheduler_stream);
        LOG_DEBUG("Set cuBLAS stream from %p to %p", original_stream, scheduler_stream);
    }
    
    switch (op->type) {
        case OperationType::CUBLAS_SGEMM: {
            auto& p = std::get<CublasGemmParams>(op->params);
            result = g_cublas_funcs.cublasSgemm_v2(
                (cublasHandle_t)p.handle,
                (cublasOperation_t)p.transa,
                (cublasOperation_t)p.transb,
                p.m, p.n, p.k,
                (const float*)p.alpha,
                (const float*)p.A, p.lda,
                (const float*)p.B, p.ldb,
                (const float*)p.beta,
                (float*)p.C, p.ldc
            );
            break;
        }
        
        case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED: {
            auto& p = std::get<CublasGemmParams>(op->params);
            result = g_cublas_funcs.cublasSgemmStridedBatched(
                (cublasHandle_t)p.handle,
                (cublasOperation_t)p.transa,
                (cublasOperation_t)p.transb,
                p.m, p.n, p.k,
                (const float*)p.alpha,
                (const float*)p.A, p.lda, p.strideA,
                (const float*)p.B, p.ldb, p.strideB,
                (const float*)p.beta,
                (float*)p.C, p.ldc, p.strideC,
                p.batchCount
            );
            break;
        }
        
        default:
            LOG_ERROR("Unknown cuBLAS operation type: %d", (int)op->type);
            result = 1;
            break;
    }
    
    // 恢复原始 stream
    if (handle && original_stream && g_cublas_funcs.cublasSetStream_v2) {
        g_cublas_funcs.cublasSetStream_v2(handle, original_stream);
    }
    
    tl_in_scheduler_execution = false;
    
    return result;
}

// 直接调用真实函数
cublasStatus_t real_cublasSgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb,
    const float* beta, float* C, int ldc) {
    
    GET_CUBLAS_FUNC(cublasSgemm_v2);
    return g_cublas_funcs.cublasSgemm_v2(
        handle, transa, transb, m, n, k,
        alpha, A, lda, B, ldb, beta, C, ldc
    );
}

cublasStatus_t real_cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const A[], int lda,
    const float* const B[], int ldb,
    const float* beta,
    float* const C[], int ldc,
    int batchCount) {
    
    GET_CUBLAS_FUNC(cublasSgemmBatched);
    return g_cublas_funcs.cublasSgemmBatched(
        handle, transa, transb, m, n, k,
        alpha, A, lda, B, ldb, beta, C, ldc, batchCount
    );
}

cublasStatus_t real_cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    const float* beta,
    float* C, int ldc, long long strideC,
    int batchCount) {
    
    GET_CUBLAS_FUNC(cublasSgemmStridedBatched);
    return g_cublas_funcs.cublasSgemmStridedBatched(
        handle, transa, transb, m, n, k,
        alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount
    );
}

} // namespace orion

// ============================================================================
// cuBLAS API Wrappers
// ============================================================================

extern "C" {

/**
 * cublasSgemm_v2 wrapper - 单精度 GEMM
 */
// 线程局部重入保护
static thread_local bool tl_in_sgemm = false;

cublasStatus_t cublasSgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb,
    const float* beta, float* C, int ldc) {
    
    using namespace orion;
    
    // 重入保护：防止递归调用
    if (tl_in_sgemm) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasSgemm_v2) {
            return g_cublas_funcs.cublasSgemm_v2(handle, transa, transb, m, n, k,
                                                  alpha, A, lda, B, ldb, beta, C, ldc);
        }
        return 1;
    }
    
    // 确保初始化
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    // 如果找不到真实函数，报错
    if (!g_cublas_funcs.cublasSgemm_v2) {
        LOG_ERROR("cublasSgemm_v2 not found");
        return 1;
    }
    
    // 调度器未初始化时直接透传
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasSgemm_v2(handle, transa, transb, m, n, k,
                                              alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    if (!is_capture_enabled()) {
        return real_cublasSgemm_v2(handle, transa, transb, m, n, k,
                                    alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return real_cublasSgemm_v2(handle, transa, transb, m, n, k,
                                    alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    // 记录拦截
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasSgemm_v2 intercepted (total: %lu)", client_idx, count + 1);
    }
    
    // 创建操作并提交到调度器队列
    auto op = create_operation(client_idx, OperationType::CUBLAS_SGEMM);
    if (!op) {
        return real_cublasSgemm_v2(handle, transa, transb, m, n, k,
                                    alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    // 设置 cuBLAS GEMM 参数
    CublasGemmParams params;
    params.handle = handle;
    params.transa = transa;
    params.transb = transb;
    params.m = m;
    params.n = n;
    params.k = k;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.B = B;
    params.ldb = ldb;
    params.beta = beta;
    params.C = C;
    params.ldc = ldc;
    params.is_batched = false;
    params.is_strided = false;
    op->params = params;
    
    // 提交到队列并等待
    enqueue_operation(op);
    wait_operation(op);
    
    return op->result == cudaSuccess ? CUBLAS_STATUS_SUCCESS : 1;
}

/**
 * cublasSgemmBatched wrapper - 批量 GEMM
 */
// 线程局部重入保护
static thread_local bool tl_in_batched = false;

cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const A[], int lda,
    const float* const B[], int ldb,
    const float* beta,
    float* const C[], int ldc,
    int batchCount) {
    
    using namespace orion;
    
    // 重入保护
    if (tl_in_batched) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasSgemmBatched) {
            return g_cublas_funcs.cublasSgemmBatched(handle, transa, transb, m, n, k,
                                                      alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
        }
        return 1;
    }
    
    // 确保初始化
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasSgemmBatched) {
        LOG_ERROR("cublasSgemmBatched not found");
        return 1;
    }
    
    // 调度器未初始化时直接透传
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasSgemmBatched(handle, transa, transb, m, n, k,
                                                  alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
    }
    
    if (!is_capture_enabled()) {
        return real_cublasSgemmBatched(handle, transa, transb, m, n, k,
                                        alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return real_cublasSgemmBatched(handle, transa, transb, m, n, k,
                                        alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
    }
    
    // 记录拦截
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasSgemmBatched intercepted (total: %lu)", client_idx, count + 1);
    }
    
    // 设置重入标记，防止内部 cudaLaunchKernel 被拦截
    tl_in_batched = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasSgemmBatched(handle, transa, transb, m, n, k,
                                                               alpha, A, lda, B, ldb, beta, C, ldc, batchCount);
    tl_in_scheduler_execution = false;
    tl_in_batched = false;
    
    return result;
}

/**
 * cublasSgemmStridedBatched wrapper
 */
// 线程局部重入保护
static thread_local bool tl_in_strided_batched = false;

cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    const float* beta,
    float* C, int ldc, long long strideC,
    int batchCount) {
    
    using namespace orion;
    
    // 重入保护
    if (tl_in_strided_batched) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasSgemmStridedBatched) {
            return g_cublas_funcs.cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                                            alpha, A, lda, strideA, B, ldb, strideB,
                                                            beta, C, ldc, strideC, batchCount);
        }
        return 1;
    }
    
    // 确保初始化
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasSgemmStridedBatched) {
        LOG_ERROR("cublasSgemmStridedBatched not found");
        return 1;
    }
    
    // 调度器未初始化时直接透传
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                                         alpha, A, lda, strideA, B, ldb, strideB,
                                                         beta, C, ldc, strideC, batchCount);
    }
    
    if (!is_capture_enabled()) {
        return real_cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                               alpha, A, lda, strideA, B, ldb, strideB,
                                               beta, C, ldc, strideC, batchCount);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return real_cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                               alpha, A, lda, strideA, B, ldb, strideB,
                                               beta, C, ldc, strideC, batchCount);
    }
    
    // 记录拦截
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasSgemmStridedBatched intercepted (total: %lu)", client_idx, count + 1);
    }
    
    // 创建操作并提交到调度器队列
    auto op = create_operation(client_idx, OperationType::CUBLAS_SGEMM_STRIDED_BATCHED);
    if (!op) {
        return real_cublasSgemmStridedBatched(handle, transa, transb, m, n, k,
                                               alpha, A, lda, strideA, B, ldb, strideB,
                                               beta, C, ldc, strideC, batchCount);
    }
    
    // 设置参数
    CublasGemmParams params;
    params.handle = handle;
    params.transa = transa;
    params.transb = transb;
    params.m = m;
    params.n = n;
    params.k = k;
    params.alpha = alpha;
    params.A = A;
    params.lda = lda;
    params.strideA = strideA;
    params.B = B;
    params.ldb = ldb;
    params.strideB = strideB;
    params.beta = beta;
    params.C = C;
    params.ldc = ldc;
    params.strideC = strideC;
    params.batchCount = batchCount;
    params.is_batched = false;
    params.is_strided = true;
    op->params = params;
    
    // 提交到队列并等待
    enqueue_operation(op);
    wait_operation(op);
    
    return op->result == cudaSuccess ? CUBLAS_STATUS_SUCCESS : 1;
}

/**
 * cublasGemmEx wrapper - 混合精度 GEMM
 */
static thread_local bool tl_in_gemmex = false;

cublasStatus_t cublasGemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const void* beta,
    void* C, int Ctype, int ldc,
    int computeType, int algo) {
    
    using namespace orion;
    
    // 重入保护
    if (tl_in_gemmex) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasGemmEx) {
            return g_cublas_funcs.cublasGemmEx(handle, transa, transb, m, n, k,
                                                alpha, A, Atype, lda, B, Btype, ldb,
                                                beta, C, Ctype, ldc, computeType, algo);
        }
        return 1;
    }
    
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasGemmEx) {
        LOG_ERROR("cublasGemmEx not found");
        return 1;
    }
    
    // 调度器未初始化时直接透传
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasGemmEx(handle, transa, transb, m, n, k,
                                            alpha, A, Atype, lda, B, Btype, ldb,
                                            beta, C, Ctype, ldc, computeType, algo);
    }
    
    if (!is_capture_enabled()) {
        return g_cublas_funcs.cublasGemmEx(handle, transa, transb, m, n, k,
                                            alpha, A, Atype, lda, B, Btype, ldb,
                                            beta, C, Ctype, ldc, computeType, algo);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return g_cublas_funcs.cublasGemmEx(handle, transa, transb, m, n, k,
                                            alpha, A, Atype, lda, B, Btype, ldb,
                                            beta, C, Ctype, ldc, computeType, algo);
    }
    
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasGemmEx intercepted (total: %lu)", client_idx, count + 1);
    }
    
    tl_in_gemmex = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasGemmEx(handle, transa, transb, m, n, k,
                                                         alpha, A, Atype, lda, B, Btype, ldb,
                                                         beta, C, Ctype, ldc, computeType, algo);
    tl_in_scheduler_execution = false;
    tl_in_gemmex = false;
    
    return result;
}

/**
 * cublasGemmStridedBatchedEx wrapper - 混合精度批量 GEMM
 */
static thread_local bool tl_in_gemm_strided_batched_ex = false;

cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, int Atype, int lda, long long strideA,
    const void* B, int Btype, int ldb, long long strideB,
    const void* beta,
    void* C, int Ctype, int ldc, long long strideC,
    int batchCount,
    int computeType, int algo) {
    
    using namespace orion;
    
    if (tl_in_gemm_strided_batched_ex) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasGemmStridedBatchedEx) {
            return g_cublas_funcs.cublasGemmStridedBatchedEx(
                handle, transa, transb, m, n, k, alpha,
                A, Atype, lda, strideA, B, Btype, ldb, strideB,
                beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
        }
        return 1;
    }
    
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasGemmStridedBatchedEx) {
        LOG_ERROR("cublasGemmStridedBatchedEx not found");
        return 1;
    }
    
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasGemmStridedBatchedEx(
            handle, transa, transb, m, n, k, alpha,
            A, Atype, lda, strideA, B, Btype, ldb, strideB,
            beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
    }
    
    if (!is_capture_enabled()) {
        return g_cublas_funcs.cublasGemmStridedBatchedEx(
            handle, transa, transb, m, n, k, alpha,
            A, Atype, lda, strideA, B, Btype, ldb, strideB,
            beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return g_cublas_funcs.cublasGemmStridedBatchedEx(
            handle, transa, transb, m, n, k, alpha,
            A, Atype, lda, strideA, B, Btype, ldb, strideB,
            beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
    }
    
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasGemmStridedBatchedEx intercepted (total: %lu)", client_idx, count + 1);
    }
    
    tl_in_gemm_strided_batched_ex = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, alpha,
        A, Atype, lda, strideA, B, Btype, ldb, strideB,
        beta, C, Ctype, ldc, strideC, batchCount, computeType, algo);
    tl_in_scheduler_execution = false;
    tl_in_gemm_strided_batched_ex = false;
    
    return result;
}

/**
 * cublasHgemm wrapper - FP16 GEMM
 */
static thread_local bool tl_in_hgemm = false;

cublasStatus_t cublasHgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, int lda,
    const void* B, int ldb,
    const void* beta,
    void* C, int ldc) {
    
    using namespace orion;
    
    if (tl_in_hgemm) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasHgemm) {
            return g_cublas_funcs.cublasHgemm(handle, transa, transb, m, n, k,
                                               alpha, A, lda, B, ldb, beta, C, ldc);
        }
        return 1;
    }
    
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasHgemm) {
        LOG_ERROR("cublasHgemm not found");
        return 1;
    }
    
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasHgemm(handle, transa, transb, m, n, k,
                                           alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    if (!is_capture_enabled()) {
        return g_cublas_funcs.cublasHgemm(handle, transa, transb, m, n, k,
                                           alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return g_cublas_funcs.cublasHgemm(handle, transa, transb, m, n, k,
                                           alpha, A, lda, B, ldb, beta, C, ldc);
    }
    
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasHgemm intercepted (total: %lu)", client_idx, count + 1);
    }
    
    tl_in_hgemm = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasHgemm(handle, transa, transb, m, n, k,
                                                        alpha, A, lda, B, ldb, beta, C, ldc);
    tl_in_scheduler_execution = false;
    tl_in_hgemm = false;
    
    return result;
}

/**
 * cublasHgemmStridedBatched wrapper - FP16 批量 GEMM
 */
static thread_local bool tl_in_hgemm_strided_batched = false;

cublasStatus_t cublasHgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,
    const void* A, int lda, long long strideA,
    const void* B, int ldb, long long strideB,
    const void* beta,
    void* C, int ldc, long long strideC,
    int batchCount) {
    
    using namespace orion;
    
    if (tl_in_hgemm_strided_batched) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasHgemmStridedBatched) {
            return g_cublas_funcs.cublasHgemmStridedBatched(
                handle, transa, transb, m, n, k, alpha,
                A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
        }
        return 1;
    }
    
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasHgemmStridedBatched) {
        LOG_ERROR("cublasHgemmStridedBatched not found");
        return 1;
    }
    
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasHgemmStridedBatched(
            handle, transa, transb, m, n, k, alpha,
            A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    }
    
    if (!is_capture_enabled()) {
        return g_cublas_funcs.cublasHgemmStridedBatched(
            handle, transa, transb, m, n, k, alpha,
            A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return g_cublas_funcs.cublasHgemmStridedBatched(
            handle, transa, transb, m, n, k, alpha,
            A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    }
    
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasHgemmStridedBatched intercepted (total: %lu)", client_idx, count + 1);
    }
    
    tl_in_hgemm_strided_batched = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasHgemmStridedBatched(
        handle, transa, transb, m, n, k, alpha,
        A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount);
    tl_in_scheduler_execution = false;
    tl_in_hgemm_strided_batched = false;
    
    return result;
}

/**
 * cublasSgemmEx wrapper - 扩展 FP32 GEMM
 */
static thread_local bool tl_in_sgemmex = false;

cublasStatus_t cublasSgemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const void* A, int Atype, int lda,
    const void* B, int Btype, int ldb,
    const float* beta,
    void* C, int Ctype, int ldc) {
    
    using namespace orion;
    
    if (tl_in_sgemmex) {
        if (!g_cublas_funcs.initialized) init_cublas_functions();
        if (g_cublas_funcs.cublasSgemmEx) {
            return g_cublas_funcs.cublasSgemmEx(handle, transa, transb, m, n, k,
                                                 alpha, A, Atype, lda, B, Btype, ldb,
                                                 beta, C, Ctype, ldc);
        }
        return 1;
    }
    
    if (!g_cublas_funcs.initialized) init_cublas_functions();
    
    if (!g_cublas_funcs.cublasSgemmEx) {
        LOG_ERROR("cublasSgemmEx not found");
        return 1;
    }
    
    if (!g_capture_state.initialized.load()) {
        return g_cublas_funcs.cublasSgemmEx(handle, transa, transb, m, n, k,
                                             alpha, A, Atype, lda, B, Btype, ldb,
                                             beta, C, Ctype, ldc);
    }
    
    if (!is_capture_enabled()) {
        return g_cublas_funcs.cublasSgemmEx(handle, transa, transb, m, n, k,
                                             alpha, A, Atype, lda, B, Btype, ldb,
                                             beta, C, Ctype, ldc);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return g_cublas_funcs.cublasSgemmEx(handle, transa, transb, m, n, k,
                                             alpha, A, Atype, lda, B, Btype, ldb,
                                             beta, C, Ctype, ldc);
    }
    
    static std::atomic<uint64_t> intercept_count{0};
    uint64_t count = intercept_count.fetch_add(1);
    if (count == 0 || (count % 100) == 0) {
        LOG_INFO("Client %d: cublasSgemmEx intercepted (total: %lu)", client_idx, count + 1);
    }
    
    tl_in_sgemmex = true;
    tl_in_scheduler_execution = true;
    cublasStatus_t result = g_cublas_funcs.cublasSgemmEx(handle, transa, transb, m, n, k,
                                                          alpha, A, Atype, lda, B, Btype, ldb,
                                                          beta, C, Ctype, ldc);
    tl_in_scheduler_execution = false;
    tl_in_sgemmex = false;
    
    return result;
}

} // extern "C"
