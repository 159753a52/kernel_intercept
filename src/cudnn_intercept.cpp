#include "gpu_capture.h"
#include <dlfcn.h>
#include <cstdio>
#include <mutex>

// cuDNN 类型定义 (避免直接依赖 cudnn.h)
typedef void* cudnnHandle_t;
typedef void* cudnnTensorDescriptor_t;
typedef void* cudnnFilterDescriptor_t;
typedef void* cudnnConvolutionDescriptor_t;
typedef void* cudnnActivationDescriptor_t;
typedef void* cudnnBatchNormMode_t;
typedef int cudnnStatus_t;
typedef int cudnnConvolutionFwdAlgo_t;
typedef int cudnnConvolutionBwdDataAlgo_t;
typedef int cudnnConvolutionBwdFilterAlgo_t;

#define CUDNN_STATUS_SUCCESS 0

namespace orion {

// ============================================================================
// cuDNN 真实函数指针类型
// ============================================================================

using cudnnConvolutionForward_t = cudnnStatus_t (*)(
    cudnnHandle_t, const void*, cudnnTensorDescriptor_t, const void*,
    cudnnFilterDescriptor_t, const void*, cudnnConvolutionDescriptor_t,
    cudnnConvolutionFwdAlgo_t, void*, size_t, const void*,
    cudnnTensorDescriptor_t, void*);

using cudnnConvolutionBackwardData_t = cudnnStatus_t (*)(
    cudnnHandle_t, const void*, cudnnFilterDescriptor_t, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnConvolutionDescriptor_t,
    cudnnConvolutionBwdDataAlgo_t, void*, size_t, const void*,
    cudnnTensorDescriptor_t, void*);

using cudnnConvolutionBackwardFilter_t = cudnnStatus_t (*)(
    cudnnHandle_t, const void*, cudnnTensorDescriptor_t, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnConvolutionDescriptor_t,
    cudnnConvolutionBwdFilterAlgo_t, void*, size_t, const void*,
    cudnnFilterDescriptor_t, void*);

using cudnnBatchNormalizationForwardTraining_t = cudnnStatus_t (*)(
    cudnnHandle_t, int, const void*, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*,
    cudnnTensorDescriptor_t, const void*, const void*,
    double, void*, void*, double, void*, void*);

using cudnnBatchNormalizationForwardInference_t = cudnnStatus_t (*)(
    cudnnHandle_t, int, const void*, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, void*,
    cudnnTensorDescriptor_t, const void*, const void*,
    const void*, const void*, double);

using cudnnBatchNormalizationBackward_t = cudnnStatus_t (*)(
    cudnnHandle_t, int, const void*, const void*, const void*, const void*,
    cudnnTensorDescriptor_t, const void*, cudnnTensorDescriptor_t, const void*,
    cudnnTensorDescriptor_t, void*, cudnnTensorDescriptor_t, const void*,
    void*, void*, double, const void*, const void*);

// ============================================================================
// 真实函数指针存储
// ============================================================================

static struct {
    cudnnConvolutionForward_t cudnnConvolutionForward;
    cudnnConvolutionBackwardData_t cudnnConvolutionBackwardData;
    cudnnConvolutionBackwardFilter_t cudnnConvolutionBackwardFilter;
    cudnnBatchNormalizationForwardTraining_t cudnnBatchNormalizationForwardTraining;
    cudnnBatchNormalizationForwardInference_t cudnnBatchNormalizationForwardInference;
    cudnnBatchNormalizationBackward_t cudnnBatchNormalizationBackward;
    bool initialized;
    std::mutex init_mutex;
} g_cudnn_funcs = {nullptr};

static void* get_cudnn_func(const char* name) {
    void* fn = dlsym(RTLD_NEXT, name);
    if (fn) return fn;
    fn = dlsym(RTLD_DEFAULT, name);
    return fn;
}

static void init_cudnn_functions() {
    std::lock_guard<std::mutex> lock(g_cudnn_funcs.init_mutex);
    if (g_cudnn_funcs.initialized) return;
    
    g_cudnn_funcs.cudnnConvolutionForward = 
        (cudnnConvolutionForward_t)get_cudnn_func("cudnnConvolutionForward");
    g_cudnn_funcs.cudnnConvolutionBackwardData = 
        (cudnnConvolutionBackwardData_t)get_cudnn_func("cudnnConvolutionBackwardData");
    g_cudnn_funcs.cudnnConvolutionBackwardFilter = 
        (cudnnConvolutionBackwardFilter_t)get_cudnn_func("cudnnConvolutionBackwardFilter");
    g_cudnn_funcs.cudnnBatchNormalizationForwardTraining = 
        (cudnnBatchNormalizationForwardTraining_t)get_cudnn_func("cudnnBatchNormalizationForwardTraining");
    g_cudnn_funcs.cudnnBatchNormalizationForwardInference = 
        (cudnnBatchNormalizationForwardInference_t)get_cudnn_func("cudnnBatchNormalizationForwardInference");
    g_cudnn_funcs.cudnnBatchNormalizationBackward = 
        (cudnnBatchNormalizationBackward_t)get_cudnn_func("cudnnBatchNormalizationBackward");
    
    g_cudnn_funcs.initialized = true;
    LOG_DEBUG("cuDNN functions initialized");
}

#define GET_CUDNN_FUNC(name) \
    do { \
        if (!g_cudnn_funcs.initialized) init_cudnn_functions(); \
        if (!g_cudnn_funcs.name) { \
            LOG_ERROR("Failed to get real " #name); \
            return 1; \
        } \
    } while(0)

// ============================================================================
// 执行真实 cuDNN 操作 (由调度器调用)
// ============================================================================

cudnnStatus_t execute_cudnn_conv_fwd(OperationPtr op) {
    GET_CUDNN_FUNC(cudnnConvolutionForward);
    auto& p = std::get<CudnnConvParams>(op->params);
    return g_cudnn_funcs.cudnnConvolutionForward(
        (cudnnHandle_t)p.handle, p.alpha, 
        (cudnnTensorDescriptor_t)p.xDesc, p.x,
        (cudnnFilterDescriptor_t)p.wDesc, p.w,
        (cudnnConvolutionDescriptor_t)p.convDesc,
        (cudnnConvolutionFwdAlgo_t)p.algo, p.workSpace, p.workSpaceSizeInBytes,
        p.beta, (cudnnTensorDescriptor_t)p.yDesc, p.y
    );
}

cudnnStatus_t execute_cudnn_operation(OperationPtr op, cudaStream_t scheduler_stream) {
    // TODO: 使用 cudnnSetStream 设置调度器的 stream
    (void)scheduler_stream;
    
    switch (op->type) {
        case OperationType::CUDNN_CONV_FWD:
            return execute_cudnn_conv_fwd(op);
        // 其他 cuDNN 操作类似处理
        default:
            LOG_ERROR("Unknown cuDNN operation type");
            return 1;
    }
}

// 直接调用真实函数
cudnnStatus_t real_cudnnConvolutionForward(
    cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnTensorDescriptor_t yDesc, void* y) {
    
    GET_CUDNN_FUNC(cudnnConvolutionForward);
    return g_cudnn_funcs.cudnnConvolutionForward(
        handle, alpha, xDesc, x, wDesc, w, convDesc,
        algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y
    );
}

cudnnStatus_t real_cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void* alpha,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnTensorDescriptor_t dxDesc, void* dx) {
    
    GET_CUDNN_FUNC(cudnnConvolutionBackwardData);
    return g_cudnn_funcs.cudnnConvolutionBackwardData(
        handle, alpha, wDesc, w, dyDesc, dy, convDesc,
        algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx
    );
}

cudnnStatus_t real_cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnFilterDescriptor_t dwDesc, void* dw) {
    
    GET_CUDNN_FUNC(cudnnConvolutionBackwardFilter);
    return g_cudnn_funcs.cudnnConvolutionBackwardFilter(
        handle, alpha, xDesc, x, dyDesc, dy, convDesc,
        algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw
    );
}

cudnnStatus_t real_cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, int mode,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void* bnScale, const void* bnBias,
    double exponentialAverageFactor,
    void* resultRunningMean, void* resultRunningVariance,
    double epsilon, void* resultSaveMean, void* resultSaveInvVariance) {
    
    GET_CUDNN_FUNC(cudnnBatchNormalizationForwardTraining);
    return g_cudnn_funcs.cudnnBatchNormalizationForwardTraining(
        handle, mode, alpha, beta, xDesc, x, yDesc, y,
        bnScaleBiasMeanVarDesc, bnScale, bnBias,
        exponentialAverageFactor, resultRunningMean, resultRunningVariance,
        epsilon, resultSaveMean, resultSaveInvVariance
    );
}

} // namespace orion

// ============================================================================
// cuDNN API Wrappers
// ============================================================================

extern "C" {

/**
 * cudnnConvolutionForward wrapper
 */
cudnnStatus_t cudnnConvolutionForward(
    cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionFwdAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnTensorDescriptor_t yDesc, void* y) {
    
    using namespace orion;
    
    if (!is_capture_enabled()) {
        return real_cudnnConvolutionForward(
            handle, alpha, xDesc, x, wDesc, w, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return real_cudnnConvolutionForward(
            handle, alpha, xDesc, x, wDesc, w, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, yDesc, y);
    }
    
    LOG_TRACE("Client %d: cudnnConvolutionForward", client_idx);
    
    auto op = submit_operation(client_idx, OperationType::CUDNN_CONV_FWD);
    if (!op) return 1;
    
    CudnnConvParams p;
    p.handle = handle;
    p.alpha = alpha;
    p.xDesc = xDesc;
    p.x = x;
    p.wDesc = wDesc;
    p.w = w;
    p.convDesc = convDesc;
    p.algo = algo;
    p.workSpace = workSpace;
    p.workSpaceSizeInBytes = workSpaceSizeInBytes;
    p.beta = beta;
    p.yDesc = yDesc;
    p.y = y;
    op->params = p;
    
    wait_operation(op);
    return op->result == cudaSuccess ? CUDNN_STATUS_SUCCESS : 1;
}

/**
 * cudnnConvolutionBackwardData wrapper
 */
cudnnStatus_t cudnnConvolutionBackwardData(
    cudnnHandle_t handle, const void* alpha,
    cudnnFilterDescriptor_t wDesc, const void* w,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdDataAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnTensorDescriptor_t dxDesc, void* dx) {
    
    using namespace orion;
    
    if (!is_capture_enabled()) {
        return real_cudnnConvolutionBackwardData(
            handle, alpha, wDesc, w, dyDesc, dy, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return real_cudnnConvolutionBackwardData(
            handle, alpha, wDesc, w, dyDesc, dy, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, dxDesc, dx);
    }
    
    LOG_TRACE("Client %d: cudnnConvolutionBackwardData", client_idx);
    
    auto op = submit_operation(client_idx, OperationType::CUDNN_CONV_BWD_DATA);
    if (!op) return 1;
    
    CudnnConvParams p;
    p.handle = handle;
    p.alpha = alpha;
    p.wDesc = wDesc;
    p.w = w;
    p.xDesc = dyDesc;
    p.x = dy;
    p.convDesc = convDesc;
    p.algo = algo;
    p.workSpace = workSpace;
    p.workSpaceSizeInBytes = workSpaceSizeInBytes;
    p.beta = beta;
    p.yDesc = dxDesc;
    p.y = dx;
    op->params = p;
    
    wait_operation(op);
    return op->result == cudaSuccess ? CUDNN_STATUS_SUCCESS : 1;
}

/**
 * cudnnConvolutionBackwardFilter wrapper
 */
cudnnStatus_t cudnnConvolutionBackwardFilter(
    cudnnHandle_t handle, const void* alpha,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t dyDesc, const void* dy,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnConvolutionBwdFilterAlgo_t algo, void* workSpace, size_t workSpaceSizeInBytes,
    const void* beta, cudnnFilterDescriptor_t dwDesc, void* dw) {
    
    using namespace orion;
    
    if (!is_capture_enabled()) {
        return real_cudnnConvolutionBackwardFilter(
            handle, alpha, xDesc, x, dyDesc, dy, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return real_cudnnConvolutionBackwardFilter(
            handle, alpha, xDesc, x, dyDesc, dy, convDesc,
            algo, workSpace, workSpaceSizeInBytes, beta, dwDesc, dw);
    }
    
    LOG_TRACE("Client %d: cudnnConvolutionBackwardFilter", client_idx);
    
    auto op = submit_operation(client_idx, OperationType::CUDNN_CONV_BWD_FILTER);
    if (!op) return 1;
    
    CudnnConvParams p;
    p.handle = handle;
    p.alpha = alpha;
    p.xDesc = xDesc;
    p.x = x;
    p.yDesc = dyDesc;
    p.y = (void*)dy;
    p.convDesc = convDesc;
    p.algo = algo;
    p.workSpace = workSpace;
    p.workSpaceSizeInBytes = workSpaceSizeInBytes;
    p.beta = beta;
    p.wDesc = dwDesc;
    p.w = dw;
    op->params = p;
    
    wait_operation(op);
    return op->result == cudaSuccess ? CUDNN_STATUS_SUCCESS : 1;
}

/**
 * cudnnBatchNormalizationForwardTraining wrapper
 */
cudnnStatus_t cudnnBatchNormalizationForwardTraining(
    cudnnHandle_t handle, int mode,
    const void* alpha, const void* beta,
    cudnnTensorDescriptor_t xDesc, const void* x,
    cudnnTensorDescriptor_t yDesc, void* y,
    cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc,
    const void* bnScale, const void* bnBias,
    double exponentialAverageFactor,
    void* resultRunningMean, void* resultRunningVariance,
    double epsilon, void* resultSaveMean, void* resultSaveInvVariance) {
    
    using namespace orion;
    
    if (!is_capture_enabled()) {
        return real_cudnnBatchNormalizationForwardTraining(
            handle, mode, alpha, beta, xDesc, x, yDesc, y,
            bnScaleBiasMeanVarDesc, bnScale, bnBias,
            exponentialAverageFactor, resultRunningMean, resultRunningVariance,
            epsilon, resultSaveMean, resultSaveInvVariance);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return real_cudnnBatchNormalizationForwardTraining(
            handle, mode, alpha, beta, xDesc, x, yDesc, y,
            bnScaleBiasMeanVarDesc, bnScale, bnBias,
            exponentialAverageFactor, resultRunningMean, resultRunningVariance,
            epsilon, resultSaveMean, resultSaveInvVariance);
    }
    
    LOG_TRACE("Client %d: cudnnBatchNormalizationForwardTraining", client_idx);
    
    auto op = submit_operation(client_idx, OperationType::CUDNN_BATCHNORM_FWD);
    if (!op) return 1;
    
    CudnnBatchNormParams p;
    p.handle = handle;
    p.mode = mode;
    p.alpha = alpha;
    p.beta = beta;
    p.xDesc = xDesc;
    p.x = x;
    p.yDesc = yDesc;
    p.y = y;
    p.bnScaleBiasMeanVarDesc = bnScaleBiasMeanVarDesc;
    p.bnScale = bnScale;
    p.bnBias = bnBias;
    p.exponentialAverageFactor = exponentialAverageFactor;
    p.resultRunningMean = resultRunningMean;
    p.resultRunningVariance = resultRunningVariance;
    p.epsilon = epsilon;
    p.resultSaveMean = resultSaveMean;
    p.resultSaveInvVariance = resultSaveInvVariance;
    op->params = p;
    
    wait_operation(op);
    return op->result == cudaSuccess ? CUDNN_STATUS_SUCCESS : 1;
}

} // extern "C"
