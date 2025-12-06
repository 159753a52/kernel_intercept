# cuBLAS 和 cuDNN 拦截分析

## 目标
理解高层库调用如何映射到统一的捕获与调度逻辑。

---

## 1. 拦截方式对比

### 1.1 三种库的拦截结构

| 库 | 拦截层级 | 是否走统一队列 |
|---|---------|---------------|
| CUDA Runtime | `cudaLaunchKernel` | ✅ 全部走队列 |
| cuBLAS | `cublasSgemm_v2` 等 | ⚠️ 部分走队列，部分透传 |
| cuDNN | `cudnnConvolutionForward` 等 | ✅ 全部走队列 |

### 1.2 为什么 cuBLAS 部分透传？

```cpp
// cublasSgemm_v2: 走队列
enqueue_operation(op);
wait_operation(op);

// cublasGemmEx: 透传（设置重入标志直接执行）
tl_in_scheduler_execution = true;
result = g_cublas_funcs.cublasGemmEx(...);
tl_in_scheduler_execution = false;
```

原因：部分 cuBLAS 函数参数复杂（如 `cublasGemmEx` 有数据类型参数），暂未实现完整的参数保存。

---

## 2. 统一拦截模式

### 2.1 cuBLAS (以 cublasSgemm_v2 为例)

```cpp
cublasStatus_t cublasSgemm_v2(handle, transa, transb, m, n, k,
                               alpha, A, lda, B, ldb, beta, C, ldc) {
    // 1. 透传检查
    if (!is_capture_enabled()) return real_cublasSgemm_v2(...);
    if (client_idx < 0) return real_cublasSgemm_v2(...);
    
    // 2. 创建操作
    auto op = create_operation(client_idx, CUBLAS_SGEMM);
    
    // 3. 保存参数
    CublasGemmParams params;
    params.handle = handle;
    params.m = m; params.n = n; params.k = k;
    params.alpha = alpha; params.beta = beta;
    params.A = A; params.B = B; params.C = C;
    params.lda = lda; params.ldb = ldb; params.ldc = ldc;
    op->params = params;
    
    // 4. 入队 + 等待（统一路径）
    enqueue_operation(op);
    wait_operation(op);
    
    return op->result;
}
```

### 2.2 cuDNN (以 cudnnConvolutionForward 为例)

```cpp
cudnnStatus_t cudnnConvolutionForward(handle, alpha, xDesc, x,
                                       wDesc, w, convDesc, algo,
                                       workspace, wsSize, beta, yDesc, y) {
    // 1. 透传检查
    if (!is_capture_enabled()) return real_cudnnConvolutionForward(...);
    if (client_idx < 0) return real_cudnnConvolutionForward(...);
    
    // 2. 创建操作 (使用 submit_operation = create + enqueue)
    auto op = submit_operation(client_idx, CUDNN_CONV_FWD);
    
    // 3. 保存参数
    CudnnConvParams p;
    p.handle = handle;
    p.xDesc = xDesc; p.x = x;
    p.wDesc = wDesc; p.w = w;
    p.convDesc = convDesc;
    p.algo = algo;
    p.workSpace = workspace;
    p.workSpaceSizeInBytes = wsSize;
    p.yDesc = yDesc; p.y = y;
    op->params = p;
    
    // 4. 等待（统一路径）
    wait_operation(op);
    
    return op->result;
}
```

---

## 3. 高层调用 → 底层 kernel 的映射

### 3.1 cuBLAS GEMM

```
应用层:   torch.mm(A, B)
           ↓
PyTorch:  at::mm → at::native::mm
           ↓
cuBLAS:   cublasSgemm_v2(handle, ...)
           ↓ 被拦截
我们:     create_operation(CUBLAS_SGEMM)
          params = {m, n, k, A, B, C, ...}
          enqueue → scheduler
           ↓
调度器:   execute_cublas_operation(op)
           ↓
真实cuBLAS: g_cublas_funcs.cublasSgemm_v2(...)
           ↓ 内部调用
CUDA:     cudaLaunchKernel (被重入保护跳过)
           ↓
GPU:      GEMM kernel 执行
```

### 3.2 cuDNN 卷积

```
应用层:   F.conv2d(input, weight)
           ↓
PyTorch:  at::conv2d → cudnn_convolution
           ↓
cuDNN:    cudnnConvolutionForward(...)
           ↓ 被拦截
我们:     create_operation(CUDNN_CONV_FWD)
          params = {xDesc, wDesc, convDesc, algo, ...}
          enqueue → scheduler
           ↓
调度器:   execute_cudnn_operation(op)
           ↓
真实cuDNN: g_cudnn_funcs.cudnnConvolutionForward(...)
           ↓
GPU:      卷积 kernel 执行
```

---

## 4. 参数结构体定义

### 4.1 CublasGemmParams

```cpp
struct CublasGemmParams {
    void* handle;
    int transa, transb;
    int m, n, k;
    const void* alpha;
    const void* A; int lda; long long strideA;
    const void* B; int ldb; long long strideB;
    const void* beta;
    void* C; int ldc; long long strideC;
    int batchCount;
    bool is_batched;
    bool is_strided;
};
```

### 4.2 CudnnConvParams

```cpp
struct CudnnConvParams {
    void* handle;
    const void* alpha;
    void* xDesc; const void* x;
    void* wDesc; const void* w;
    void* convDesc;
    int algo;
    void* workSpace;
    size_t workSpaceSizeInBytes;
    const void* beta;
    void* yDesc; void* y;
};
```

---

## 5. 调度器执行函数

### 5.1 cuBLAS 执行

```cpp
cublasStatus_t execute_cublas_operation(OperationPtr op, cudaStream_t stream) {
    tl_in_scheduler_execution = true;  // 防止内部 cudaLaunchKernel 被拦截
    
    // 设置调度器的 stream
    cublasSetStream_v2(handle, stream);
    
    switch (op->type) {
        case CUBLAS_SGEMM:
            result = g_cublas_funcs.cublasSgemm_v2(...);
            break;
        case CUBLAS_SGEMM_STRIDED_BATCHED:
            result = g_cublas_funcs.cublasSgemmStridedBatched(...);
            break;
    }
    
    tl_in_scheduler_execution = false;
    return result;
}
```

### 5.2 cuDNN 执行

```cpp
cudnnStatus_t execute_cudnn_operation(OperationPtr op, cudaStream_t stream) {
    // TODO: cudnnSetStream(handle, stream)
    
    switch (op->type) {
        case CUDNN_CONV_FWD:
            return execute_cudnn_conv_fwd(op);
        // ...
    }
}
```

---

## 6. 拦截的函数列表

### 6.1 cuBLAS

| 函数 | 调度方式 | 说明 |
|------|---------|------|
| `cublasSgemm_v2` | 队列调度 | FP32 GEMM |
| `cublasSgemmStridedBatched` | 队列调度 | FP32 批量 GEMM |
| `cublasSgemmBatched` | 透传 | 指针数组复杂 |
| `cublasGemmEx` | 透传 | 混合精度，参数复杂 |
| `cublasGemmStridedBatchedEx` | 透传 | 混合精度批量 |
| `cublasHgemm` | 透传 | FP16 GEMM |
| `cublasHgemmStridedBatched` | 透传 | FP16 批量 |
| `cublasSgemmEx` | 透传 | 扩展 FP32 |

### 6.2 cuDNN

| 函数 | 调度方式 | 说明 |
|------|---------|------|
| `cudnnConvolutionForward` | 队列调度 | 前向卷积 |
| `cudnnConvolutionBackwardData` | 队列调度 | 反向数据 |
| `cudnnConvolutionBackwardFilter` | 队列调度 | 反向权重 |
| `cudnnBatchNormalizationForwardTraining` | 队列调度 | BN 前向训练 |

---

## 7. 如何支持新库（如 NCCL）

### 7.1 添加步骤

```
1. 创建 src/nccl_intercept.cpp

2. 定义函数指针类型
   using ncclAllReduce_t = ncclResult_t (*)(const void*, void*, size_t, ...);

3. 定义参数结构体
   struct NcclAllReduceParams {
       const void* sendbuff;
       void* recvbuff;
       size_t count;
       ncclDataType_t datatype;
       ncclRedOp_t op;
       ncclComm_t comm;
       cudaStream_t stream;
   };

4. 在 gpu_capture.h 添加
   - OperationType::NCCL_ALLREDUCE
   - variant 中添加 NcclAllReduceParams

5. 实现 wrapper 函数
   extern "C" ncclResult_t ncclAllReduce(...) {
       auto op = create_operation(client_idx, NCCL_ALLREDUCE);
       op->params = NcclAllReduceParams{...};
       enqueue_operation(op);
       wait_operation(op);
       return op->result;
   }

6. 实现执行函数
   ncclResult_t execute_nccl_operation(OperationPtr op, cudaStream_t stream);

7. 在 scheduler.cpp 添加 case 分支
```

### 7.2 代码结构图

```
┌─────────────────────────────────────────────────────────────────┐
│                     新库拦截层模板                               │
│                                                                 │
│  src/xxx_intercept.cpp                                          │
│  ├── 函数指针类型定义 (using xxx_t = ...)                       │
│  ├── 真实函数指针存储 (g_xxx_funcs)                            │
│  ├── 初始化函数 (init_xxx_functions)                           │
│  ├── 执行函数 (execute_xxx_operation)                          │
│  └── extern "C" wrapper 函数                                    │
│                                                                 │
│  include/gpu_capture.h                                          │
│  ├── OperationType 枚举添加新类型                               │
│  ├── XxxParams 参数结构体                                       │
│  └── variant 添加新参数类型                                     │
│                                                                 │
│  src/scheduler.cpp                                              │
│  └── execute_operation() switch 添加 case                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 总结

| 方面 | 说明 |
|------|------|
| 统一性 | cuBLAS/cuDNN 都走 `enqueue → wait → execute` 路径 |
| 参数保存 | 每种操作有专门的 Params 结构体 |
| Stream 控制 | 调度器执行时设置自己的 stream |
| 重入保护 | 防止库内部 `cudaLaunchKernel` 被递归拦截 |
| 扩展性 | 新库按模板添加 `xxx_intercept.cpp` |
