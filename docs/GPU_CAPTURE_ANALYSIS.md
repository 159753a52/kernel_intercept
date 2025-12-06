# GPU 操作捕获层分析

## 目标
理解项目如何以统一方式在 C++ 层记录来自 CUDA/cuBLAS/cuDNN 的算子调用。

---

## 1. 核心数据结构：OperationRecord

**文件**: `include/gpu_capture.h` (第 150-200 行)

```cpp
struct OperationRecord {
    // ========== 基本信息 ==========
    OperationType type;      // 操作类型（枚举）
    uint64_t op_id;          // 唯一操作ID
    int client_idx;          // 所属客户端索引
    
    // ========== 操作参数 (使用 variant 统一存储) ==========
    std::variant<
        KernelLaunchParams,      // CUDA kernel 启动参数
        MallocParams,            // 内存分配参数
        FreeParams,              // 内存释放参数
        MemcpyParams,            // 内存拷贝参数
        MemsetParams,            // 内存设置参数
        SyncParams,              // 同步参数
        CudnnConvParams,         // cuDNN 卷积参数
        CudnnBatchNormParams,    // cuDNN BatchNorm 参数
        CublasGemmParams         // cuBLAS GEMM 参数
    > params;
    
    // ========== 执行状态 ==========
    std::atomic<bool> completed{false};  // 是否完成
    std::atomic<bool> started{false};    // 是否开始执行
    cudaError_t result;                  // 执行结果
    void* result_ptr;                    // malloc 的返回指针
    
    // ========== 同步机制 ==========
    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    
    // 等待操作完成
    void wait_completion();
    
    // 标记操作完成
    void mark_completed(cudaError_t res);
};
```

### 各类操作的参数结构

| 操作类型 | 参数结构体 | 关键字段 |
|---------|-----------|---------|
| KERNEL_LAUNCH | `KernelLaunchParams` | func, gridDim, blockDim, args |
| MALLOC | `MallocParams` | devPtr, size |
| FREE | `FreeParams` | devPtr |
| MEMCPY | `MemcpyParams` | dst, src, count, kind |
| CUBLAS_SGEMM | `CublasGemmParams` | m, n, k, A, B, C, alpha, beta |
| CUDNN_CONV | `CudnnConvParams` | x, w, y, algo, workspace |

---

## 2. 队列接口：ClientQueue

**文件**: `include/gpu_capture.h` (第 210-260 行)

```cpp
class ClientQueue {
private:
    std::queue<OperationPtr> queue_;    // 底层队列
    std::mutex mutex_;                   // 保护队列的互斥锁
    std::condition_variable cv_;         // 用于阻塞等待
    std::atomic<bool> shutdown_;         // 关闭标志
    
public:
    // ========== 入队操作 ==========
    void push(OperationPtr op);
    // 加锁 → 放入队列 → 解锁 → 通知等待者
    
    // ========== 出队操作 ==========
    OperationPtr try_pop();      // 非阻塞，队列空返回 nullptr
    OperationPtr wait_pop();     // 阻塞等待，直到有元素或 shutdown
    OperationPtr peek();         // 查看队首，不移除
    
    // ========== 状态查询 ==========
    bool empty();
    size_t size();
    
    // ========== 生命周期 ==========
    void shutdown();             // 通知所有等待者退出
};
```

### 队列操作流程

```
push(op):
    lock(mutex_)
    queue_.push(op)
    unlock(mutex_)
    cv_.notify_one()  ──→ 唤醒一个等待的调度器线程

try_pop():
    lock(mutex_)
    if queue_.empty() return nullptr
    op = queue_.front()
    queue_.pop()
    return op

wait_pop():
    lock(mutex_)
    cv_.wait(mutex_, [&]{ return !queue_.empty() || shutdown_; })
    if (shutdown_ && queue_.empty()) return nullptr
    op = queue_.front()
    queue_.pop()
    return op
```

---

## 3. 全局状态：CaptureLayerState

**文件**: `include/gpu_capture.h` (第 270-300 行)

```cpp
struct CaptureLayerState {
    std::atomic<bool> initialized{false};    // 是否已初始化
    std::atomic<bool> enabled{false};        // 是否启用拦截
    
    int num_clients{0};                      // 客户端数量
    
    // 每个客户端一个队列
    std::vector<std::unique_ptr<ClientQueue>> client_queues;
    
    // 操作ID生成器（原子递增）
    std::atomic<uint64_t> next_op_id{0};
    
    // 调度器通知
    std::mutex scheduler_mutex;
    std::condition_variable scheduler_cv;
};

// 全局唯一实例
extern CaptureLayerState g_capture_state;
```

---

## 4. 拦截层交互 API

**文件**: `src/gpu_capture.cpp`

### 4.1 初始化/关闭

```cpp
// 初始化拦截层，创建 num_clients 个队列
int init_capture_layer(int num_clients);

// 关闭拦截层，通知所有队列 shutdown
void shutdown_capture_layer();
```

### 4.2 客户端管理

```cpp
// 线程局部变量，存储当前线程的客户端索引
thread_local int tl_client_idx = -1;

// 获取/设置当前线程的客户端索引
int get_current_client_idx();
void set_current_client_idx(int idx);

// 检查当前线程是否被管理
bool is_managed_thread();

// 检查拦截是否启用
bool is_capture_enabled();
```

### 4.3 操作提交（核心 API）

```cpp
// 创建操作记录（不入队）
OperationPtr create_operation(int client_idx, OperationType type);

// 将操作放入对应客户端的队列
void enqueue_operation(OperationPtr op);

// 等待操作完成
void wait_operation(OperationPtr op);

// 通知调度器有新操作
void notify_scheduler();
```

---

## 5. 时序图：拦截函数调用流程

以 `cudaMalloc` 为例：

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   客户端线程     │     │   gpu_capture   │     │   调度器线程     │
│  (Python/App)   │     │                 │     │                 │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │  cudaMalloc(ptr, sz)  │                       │
         │──────────────────────>│                       │
         │                       │                       │
         │  (1) 检查拦截状态      │                       │
         │  is_capture_enabled() │                       │
         │                       │                       │
         │  (2) 获取客户端索引    │                       │
         │  get_current_client_idx()                     │
         │                       │                       │
         │  (3) 创建操作记录      │                       │
         │  create_operation(idx, MALLOC)                │
         │                       │                       │
         │  (4) 设置参数          │                       │
         │  op->params = MallocParams{ptr, sz}           │
         │                       │                       │
         │  (5) 入队              │                       │
         │  enqueue_operation(op)│                       │
         │                       │──────────────────────>│
         │                       │  ClientQueue::push()  │
         │                       │  notify_scheduler()   │
         │                       │                       │
         │  (6) 等待完成          │                       │
         │  wait_operation(op)   │                       │
         │       ┌───────────────┤                       │
         │       │ cv.wait()     │                       │
         │       │ (阻塞)        │  (7) 取出操作         │
         │       │               │<──────────────────────│
         │       │               │  try_pop() / wait_pop()
         │       │               │                       │
         │       │               │  (8) 执行真实 CUDA    │
         │       │               │  real_cudaMalloc()    │
         │       │               │                       │
         │       │               │  (9) 标记完成         │
         │       │               │  op->mark_completed() │
         │       │               │──────────────────────>│
         │       │               │  cv.notify_all()      │
         │       └───────────────┤                       │
         │  (10) 返回结果        │                       │
         │<──────────────────────│                       │
         │  return op->result    │                       │
         │                       │                       │
```

---

## 6. 各拦截文件的调用模式

### 6.1 CUDA Runtime (`cuda_intercept.cpp`)

```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size) {
    // 透传检查
    SAFE_PASSTHROUGH(cudaMalloc, devPtr, size);
    if (!is_capture_enabled()) return real_cudaMalloc(devPtr, size);
    int client_idx = get_current_client_idx();
    if (client_idx < 0) return real_cudaMalloc(devPtr, size);
    
    // 创建 → 设置参数 → 入队 → 等待
    auto op = create_operation(client_idx, OperationType::MALLOC);
    op->params = MallocParams{devPtr, size};
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}
```

### 6.2 cuBLAS (`cublas_intercept.cpp`)

```cpp
cublasStatus_t cublasSgemm_v2(...) {
    // 透传检查（类似）
    
    // 创建 → 设置参数 → 入队 → 等待
    auto op = create_operation(client_idx, OperationType::CUBLAS_SGEMM);
    CublasGemmParams params;
    params.handle = handle;
    params.m = m; params.n = n; params.k = k;
    // ... 设置其他参数
    op->params = params;
    enqueue_operation(op);
    wait_operation(op);
    return (cublasStatus_t)op->result;
}
```

### 6.3 cuDNN (`cudnn_intercept.cpp`)

```cpp
cudnnStatus_t cudnnConvolutionForward(...) {
    // 使用旧接口（submit_operation = create + enqueue）
    auto op = submit_operation(client_idx, OperationType::CUDNN_CONV_FWD);
    // 设置参数
    // ...
    wait_operation(op);
    return (cudnnStatus_t)op->result;
}
```

---

## 7. 统一设计模式

所有拦截函数遵循相同的模式：

```cpp
ReturnType intercepted_function(args...) {
    // ======== 阶段 1: 透传检查 ========
    if (!initialized || in_scheduler_execution) {
        return real_function(args...);
    }
    if (!is_capture_enabled()) {
        return real_function(args...);
    }
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return real_function(args...);
    }
    
    // ======== 阶段 2: 创建操作记录 ========
    auto op = create_operation(client_idx, OPERATION_TYPE);
    if (!op) return real_function(args...);
    
    // ======== 阶段 3: 设置参数 ========
    XxxParams params;
    params.field1 = arg1;
    params.field2 = arg2;
    // ...
    op->params = params;
    
    // ======== 阶段 4: 入队 ========
    enqueue_operation(op);
    
    // ======== 阶段 5: 等待完成 ========
    wait_operation(op);
    
    // ======== 阶段 6: 返回结果 ========
    return (ReturnType)op->result;
}
```

---

## 8. 关键设计点

### 8.1 使用 `std::variant` 统一参数存储

- 避免了 `union` 的内存安全问题
- 支持类型安全的访问 (`std::get<T>`)
- 编译时类型检查

### 8.2 使用 `std::shared_ptr` 管理操作记录

- 客户端线程和调度器线程共享同一个 `OperationRecord`
- 自动内存管理，避免手动 delete

### 8.3 使用 `thread_local` 存储客户端索引

- 每个线程独立的客户端标识
- 无需在每次调用时传递客户端索引

### 8.4 使用条件变量实现等待/通知

- `OperationRecord::completion_cv` 用于等待操作完成
- `ClientQueue::cv_` 用于等待新操作
- `g_capture_state.scheduler_cv` 用于通知调度器

---

## 9. 总结

```
                    ┌─────────────────────────────────────────┐
                    │         统一操作记录 (OperationRecord)   │
                    │  ┌─────────────────────────────────────┐│
                    │  │ type: OperationType                 ││
                    │  │ params: variant<Kernel, Malloc,...> ││
                    │  │ completed: atomic<bool>             ││
                    │  │ result: cudaError_t                 ││
                    │  └─────────────────────────────────────┘│
                    └─────────────────────────────────────────┘
                                        ▲
                                        │ 创建
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
┌───────┴───────┐             ┌─────────┴─────────┐           ┌─────────┴─────────┐
│cuda_intercept │             │cublas_intercept   │           │cudnn_intercept    │
│               │             │                   │           │                   │
│cudaMalloc     │             │cublasSgemm        │           │cudnnConvFwd       │
│cudaLaunchKern │             │cublasSgemmBatched │           │cudnnBatchNorm     │
│cudaMemcpy     │             │...                │           │...                │
└───────────────┘             └───────────────────┘           └───────────────────┘
        │                               │                               │
        └───────────────────────────────┼───────────────────────────────┘
                                        │ enqueue
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │            ClientQueue                   │
                    │  push() / try_pop() / wait_pop()        │
                    └─────────────────────────────────────────┘
                                        │
                                        │ notify
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │            Scheduler                     │
                    │  取出操作 → 执行 → mark_completed()     │
                    └─────────────────────────────────────────┘
```
