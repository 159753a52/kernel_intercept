# Orion-Style Operator-Level GPU Scheduler Design

## 1. 整体架构

```
+------------------+     +------------------+     +------------------+
|  HP Client (0)   |     |  BE Client (1)   |     |  BE Client (N)   |
|  (High Priority) |     |  (Best Effort)   |     |  (Best Effort)   |
+--------+---------+     +--------+---------+     +--------+---------+
         |                        |                        |
         | cudaLaunchKernel       | cudaMemcpy             | cuBLAS...
         v                        v                        v
+------------------------------------------------------------------------+
|                    拦截层 (Capture Layer)                               |
|  - LD_PRELOAD + dlsym(RTLD_NEXT, ...) 拦截 CUDA/cuDNN/cuBLAS           |
|  - 封装为 OperationRecord，写入 per-client 队列                         |
|  - block_until_allowed() 等待调度器许可                                 |
+------------------------------------------------------------------------+
         |                        |                        |
         v                        v                        v
+------------------------------------------------------------------------+
|                    Per-Client 软件队列                                  |
|  Queue[0]: HP操作队列    Queue[1]: BE队列    ...    Queue[N]: BE队列   |
|  (lock-free or mutex)                                                   |
+------------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------------+
|                    调度层 (Scheduler)                                   |
|  - 单独线程，轮询所有队列                                                |
|  - HP操作优先执行                                                       |
|  - BE操作仅在满足条件时并发执行                                          |
|  - 维护 CUDA streams (HP stream + BE streams)                           |
|  - 使用 cudaEvent 跟踪完成状态                                          |
+------------------------------------------------------------------------+
         |
         v
+------------------------------------------------------------------------+
|                    GPU Hardware                                         |
|  - HP Stream (高优先级 CUDA stream)                                     |
|  - BE Streams (默认优先级 streams)                                      |
+------------------------------------------------------------------------+
```

## 2. 模块划分

### 2.1 拦截层 (capture layer)
- `gpu_capture.h/cpp`: 核心数据结构、队列、同步原语
- `cuda_intercept.cpp`: CUDA Runtime API 拦截
- `cudnn_intercept.cpp`: cuDNN API 拦截
- `cublas_intercept.cpp`: cuBLAS API 拦截

### 2.2 调度层 (scheduler)
- `scheduler.h/cpp`: Scheduler 类，调度策略，stream 管理

### 2.3 Profiling 层
- `kernel_profile.h/cpp`: Profile 加载和查询

### 2.4 Python 集成
- `launch_jobs.py`: 启动脚本

## 3. 核心数据结构

### 3.1 OperationType 枚举
```cpp
enum class OperationType {
    KERNEL_LAUNCH,
    MALLOC, FREE,
    MEMCPY, MEMCPY_ASYNC,
    MEMSET, MEMSET_ASYNC,
    CUDNN_CONV_FWD, CUDNN_CONV_BWD_DATA, CUDNN_CONV_BWD_FILTER,
    CUDNN_BATCHNORM_FWD, CUDNN_BATCHNORM_BWD,
    CUBLAS_SGEMM, CUBLAS_SGEMM_BATCHED,
    SYNC, STREAM_SYNC,
    UNKNOWN
};
```

### 3.2 OperationRecord 结构
```cpp
struct OperationRecord {
    OperationType type;
    uint64_t op_id;           // 唯一标识
    int client_idx;           // 所属 client
    
    // 参数 union 或 variant
    union Params { ... };
    
    // Profiling 信息 (可选)
    std::string kernel_id;
    float estimated_duration; // ms
    int sm_needed;
    ProfileType profile_type; // COMPUTE_BOUND, MEMORY_BOUND, UNKNOWN
    
    // 执行结果
    std::atomic<bool> completed{false};
    cudaError_t result;
    void* result_ptr;         // for malloc
};
```

## 4. 调度策略 (Orion 风格)

### 4.1 优先级模型
- Client 0: High Priority (HP)
- Client 1..N-1: Best Effort (BE)

### 4.2 调度规则
1. HP 操作永远优先执行
2. BE 操作仅在以下条件满足时并发执行:
   - `hp_task_running == false`，或
   - `op_be.sm_needed < SM_THRESHOLD` 且
   - `profile_type` 互补 (compute + memory) 且
   - `cumulative_be_duration < DUR_THRESHOLD * hp_request_latency`

### 4.3 参数
- `SM_THRESHOLD`: BE kernel 的最大 SM 占用 (默认: GPU SM 的 50%)
- `DUR_THRESHOLD`: BE kernel 累计执行时间上限比例 (默认: 2.5%)

## 5. 时序图

### 5.1 cudaMalloc 调用流程
```
Client Thread                    Scheduler Thread                GPU
    |                                 |                           |
    | cudaMalloc(ptr, size)           |                           |
    v                                 |                           |
 [Wrapper intercepts]                 |                           |
    |                                 |                           |
    | 1. Check if managed client      |                           |
    |    (YES)                        |                           |
    |                                 |                           |
    | 2. Create OperationRecord       |                           |
    |    (type=MALLOC, size, ptr_ref) |                           |
    |                                 |                           |
    | 3. Push to Queue[client_idx]    |                           |
    |                                 |                           |
    | 4. block_until_allowed()  ----->|                           |
    |    (waiting...)                 |                           |
    |                                 | [Scheduler picks up op]   |
    |                                 |                           |
    |                                 | 5. Real cudaMalloc() ---->| [Execute]
    |                                 |                           |
    |                                 | 6. Store result in op     |
    |                                 |                           |
    |                                 | 7. Set op.completed=true  |
    |                                 |    Signal condition_var   |
    |                                 |                           |
    | <------------------------------ |                           |
    | (unblock)                       |                           |
    |                                 |                           |
    | 8. Return result                |                           |
    v                                 |                           |
```

## 6. 同步语义保证

### 6.1 显式同步 (cudaDeviceSynchronize, cudaStreamSynchronize)
- 直接拦截并通过队列调度
- 调度器执行时会等待该 client 之前所有操作完成
- 保证调用返回时 GPU 状态一致

### 6.2 隐式同步 (cudaMalloc, cudaFree, cudaMemcpy)
- 这些 API 本身有隐式同步语义
- 拦截层通过 block_until_allowed() 串行化
- 调度器执行时调用真实 API，保持原始语义

### 6.3 死锁预防
1. Scheduler 线程永不等待 client 标志
2. Client 标志由 scheduler 单向设置
3. 停机流程: 设置 shutdown 标志，唤醒所有等待的 client

## 7. 性能优化

### 7.1 减少锁争用
- 使用 lock-free MPSC 队列 (多生产者单消费者)
- 或使用 per-client mutex，scheduler 依次获取

### 7.2 减少轮询开销
- Scheduler 使用条件变量等待新操作
- 有新操作时 client 通知 scheduler

### 7.3 批量调度
- 当没有 HP 操作时，一次性调度多个 BE 操作
