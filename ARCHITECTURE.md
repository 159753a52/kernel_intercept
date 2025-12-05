# 项目架构详解

## 一、整体流程

```
用户代码: torch.mm(A, B)
    │
    ▼
PyTorch 内部调用 cublasSgemm_v2()
    │
    ▼
LD_PRELOAD 使我们的 cublasSgemm_v2() 先被调用 (cublas_intercept.cpp)
    │
    ▼
创建 OperationRecord，放入队列，等待完成
    │
    ▼
Scheduler 线程从队列取出，调用真正的 cublasSgemm_v2()
    │
    ▼
执行完成，通知客户端，客户端继续
```

## 二、核心文件详解

### 1. include/common.h - 公共定义

```cpp
// 日志宏
#define LOG_ERROR(fmt, ...) ...
#define LOG_INFO(fmt, ...)  ...
#define LOG_DEBUG(fmt, ...) ...

// 操作类型枚举
enum class OperationType {
    KERNEL_LAUNCH,      // cudaLaunchKernel
    MALLOC,             // cudaMalloc
    FREE,               // cudaFree
    MEMCPY,             // cudaMemcpy
    CUBLAS_SGEMM,       // cublasSgemm
    ...
};

// Kernel 类型 (用于调度决策)
enum class ProfileType {
    COMPUTE_BOUND,      // 计算密集型
    MEMORY_BOUND,       // 带宽密集型
    UNKNOWN
};
```

### 2. include/gpu_capture.h - 拦截层接口

**核心数据结构：OperationRecord**

```cpp
struct OperationRecord {
    OperationType type;           // 操作类型
    uint64_t op_id;               // 唯一 ID
    int client_idx;               // 客户端 ID (0=HP, 1+=BE)
    
    // 操作参数 (用 variant 存储不同类型)
    std::variant<
        KernelLaunchParams,       // cudaLaunchKernel 参数
        MallocParams,             // cudaMalloc 参数
        CublasGemmParams,         // cuBLAS GEMM 参数
        ...
    > params;
    
    // 同步机制
    std::atomic<bool> completed;  // 是否完成
    std::mutex completion_mutex;
    std::condition_variable completion_cv;
    cudaError_t result;           // 执行结果
    
    void wait_completion();       // 客户端调用，等待完成
    void mark_completed(err);     // 调度器调用，标记完成
};
```

**客户端队列：**

```cpp
class ClientQueue {
    std::queue<OperationPtr> queue_;
    std::mutex mutex_;
    
    void push(OperationPtr op);   // 入队
    OperationPtr try_pop();       // 出队 (非阻塞)
    bool empty();
};
```

**全局状态：**

```cpp
struct CaptureState {
    std::vector<std::unique_ptr<ClientQueue>> client_queues;  // 每个客户端一个队列
    std::mutex scheduler_mutex;
    std::condition_variable scheduler_cv;  // 唤醒调度器
    thread_local int tl_client_idx;        // 当前线程的客户端 ID
};
```

### 3. src/gpu_capture.cpp - 拦截层实现

**关键函数：**

```cpp
// 初始化拦截层
int init_capture_layer(int num_clients) {
    g_capture_state.client_queues.resize(num_clients);
    for (int i = 0; i < num_clients; i++) {
        g_capture_state.client_queues[i] = std::make_unique<ClientQueue>();
    }
}

// 创建操作记录
OperationPtr create_operation(int client_idx, OperationType type) {
    auto op = std::make_shared<OperationRecord>();
    op->type = type;
    op->client_idx = client_idx;
    op->op_id = g_op_counter++;
    return op;
}

// 入队操作
void enqueue_operation(OperationPtr op) {
    g_capture_state.client_queues[op->client_idx]->push(op);
    g_capture_state.scheduler_cv.notify_one();  // 唤醒调度器
}

// 等待操作完成
void wait_operation(OperationPtr op) {
    op->wait_completion();  // 阻塞直到调度器执行完成
}
```

### 4. src/cuda_intercept.cpp - CUDA API 拦截

**拦截原理：LD_PRELOAD**

```cpp
// 真正的 CUDA 函数指针
static cudaError_t (*real_cudaLaunchKernel)(...) = nullptr;

// 初始化时获取真正的函数
void init_cuda_functions() {
    void* cuda_lib = dlopen("libcudart.so", RTLD_NOW);
    real_cudaLaunchKernel = dlsym(cuda_lib, "cudaLaunchKernel");
}

// 我们的拦截函数 (同名，会被优先调用)
extern "C" cudaError_t cudaLaunchKernel(
    const void* func,
    dim3 gridDim, dim3 blockDim,
    void** args,
    size_t sharedMem,
    cudaStream_t stream
) {
    // 1. 检查是否需要拦截
    int client_idx = get_client_idx();
    if (client_idx < 0) {
        // Passthrough 模式：直接调用真正的函数
        return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }
    
    // 2. 创建操作记录
    auto op = create_operation(client_idx, OperationType::KERNEL_LAUNCH);
    
    // 3. 保存参数
    KernelLaunchParams params;
    params.func = func;
    params.gridDim = gridDim;
    params.blockDim = blockDim;
    params.original_args = args;  // 保存 args 指针
    params.sharedMem = sharedMem;
    params.stream = stream;
    op->params = params;
    
    // 4. 入队并等待
    enqueue_operation(op);
    wait_operation(op);  // 阻塞！直到调度器执行完成
    
    return op->result;
}
```

**执行函数 (调度器调用)：**

```cpp
cudaError_t execute_cuda_operation(OperationPtr op, cudaStream_t scheduler_stream) {
    switch (op->type) {
        case OperationType::KERNEL_LAUNCH: {
            auto& p = std::get<KernelLaunchParams>(op->params);
            // 使用调度器分配的 stream，而不是客户端的 stream
            return real_cudaLaunchKernel(
                p.func, p.gridDim, p.blockDim,
                p.get_args(), p.sharedMem,
                scheduler_stream  // 关键：用调度器的 stream
            );
        }
        case OperationType::MALLOC: {
            auto& p = std::get<MallocParams>(op->params);
            return real_cudaMalloc(p.devPtr, p.size);
        }
        // ... 其他操作类型
    }
}
```

### 5. src/cublas_intercept.cpp - cuBLAS 拦截

与 cuda_intercept.cpp 类似：

```cpp
// 真正的 cuBLAS 函数
static cublasStatus_t (*real_cublasSgemm_v2)(...) = nullptr;

// 拦截函数
extern "C" cublasStatus_t cublasSgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc
) {
    int client_idx = get_client_idx();
    if (client_idx < 0) {
        return real_cublasSgemm_v2(...);  // Passthrough
    }
    
    // 创建操作记录
    auto op = create_operation(client_idx, OperationType::CUBLAS_SGEMM);
    
    // 保存参数
    CublasGemmParams params;
    params.handle = handle;
    params.m = m; params.n = n; params.k = k;
    params.A = A; params.B = B; params.C = C;
    // ... 保存所有参数
    op->params = params;
    
    // 入队并等待
    enqueue_operation(op);
    wait_operation(op);
    
    return op->result == cudaSuccess ? CUBLAS_STATUS_SUCCESS : 1;
}
```

### 6. src/scheduler.cpp - 调度器实现

**多线程调度器：每个客户端一个线程**

```cpp
void Scheduler::start() {
    running_ = true;
    // 为每个客户端启动一个调度器线程
    for (int i = 0; i < num_clients_; i++) {
        threads_.emplace_back(&Scheduler::run_client, this, i);
    }
}

void Scheduler::run_client(int client_idx) {
    bool is_hp = (client_idx == 0);
    cudaStream_t my_stream = is_hp ? hp_stream_ : be_streams_[client_idx - 1];
    
    while (running_) {
        // 从自己的队列取操作
        OperationPtr op = client_queues[client_idx]->try_pop();
        
        if (op) {
            if (is_hp) {
                // HP 直接执行
                execute_operation(op, my_stream);
                op->mark_completed(err);
            } else {
                // BE 需要调度决策
                if (schedule_be(current_hp_op_, op)) {
                    execute_operation(op, my_stream);
                    op->mark_completed(err);
                }
            }
        } else {
            // 等待新操作
            scheduler_cv.wait_for(lock, 10us, ...);
        }
    }
}

// 调度决策
bool Scheduler::schedule_be(OperationPtr hp_op, OperationPtr be_op) {
    // 1. 没有 HP 运行时，允许
    if (!hp_task_running_) return true;
    
    // 2. 检查 SM 占用
    if (be_op->sm_needed >= sm_threshold) return false;
    
    // 3. 检查是否互补 (compute + memory)
    if (!is_complementary(hp_op->profile_type, be_op->profile_type))
        return false;
    
    return true;
}
```

## 三、关键流程图解

### 1. 拦截和调度流程

```
Client Thread                    Scheduler Thread
     │                                 │
     │  torch.mm(A, B)                 │
     │       │                         │
     │       ▼                         │
     │  cublasSgemm_v2() [拦截]        │
     │       │                         │
     │  create_operation()             │
     │  enqueue_operation() ──────────►│ try_pop()
     │       │                         │       │
     │  wait_operation() [阻塞]        │  execute_operation()
     │       │                         │       │
     │       │◄──────────────────────────── mark_completed()
     │       │                         │
     │  return result                  │
     ▼                                 ▼
```

### 2. Stream 分配

```
Client 0 (HP) ──► Scheduler Thread 0 ──► hp_stream_ (高优先级)
                                              │
                                              ▼
                                            GPU
                                              ▲
Client 1 (BE) ──► Scheduler Thread 1 ──► be_stream_[0] (低优先级)
```

## 四、如何修改代码

### 1. 添加新的 CUDA API 拦截

在 `cuda_intercept.cpp` 中：

```cpp
// 1. 添加真正的函数指针
static cudaError_t (*real_cudaNewAPI)(...) = nullptr;

// 2. 在 init_cuda_functions() 中获取
real_cudaNewAPI = dlsym(cuda_lib, "cudaNewAPI");

// 3. 添加拦截函数
extern "C" cudaError_t cudaNewAPI(...) {
    int client_idx = get_client_idx();
    if (client_idx < 0) return real_cudaNewAPI(...);
    
    auto op = create_operation(client_idx, OperationType::NEW_API);
    // 保存参数...
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}

// 4. 在 execute_cuda_operation() 中添加执行逻辑
case OperationType::NEW_API:
    return real_cudaNewAPI(...);
```

### 2. 修改调度策略

在 `scheduler.cpp` 的 `schedule_be()` 中：

```cpp
bool Scheduler::schedule_be(OperationPtr hp_op, OperationPtr be_op) {
    // 你的调度逻辑
    // 例如：只允许 memory-bound 的 BE 操作并发
    if (be_op->profile_type == ProfileType::MEMORY_BOUND) {
        return true;
    }
    return false;
}
```

### 3. 修改开销

开销主要来自 `wait_operation()` 中的条件变量。如果要减少开销：

```cpp
// 方案1: 改用 spinlock
void wait_operation(OperationPtr op) {
    while (!op->completed.load()) {
        std::this_thread::yield();  // 忙等待
    }
}

// 方案2: 批量提交
void batch_enqueue(std::vector<OperationPtr>& ops) {
    for (auto& op : ops) {
        enqueue_operation(op);
    }
    // 只在最后一个操作等待
    wait_operation(ops.back());
}
```

## 五、调试技巧

### 1. 查看日志

```bash
# 设置日志级别 (1=ERROR, 2=WARN, 3=INFO, 4=DEBUG)
ORION_LOG_LEVEL=4 LD_PRELOAD=./build/libgpu_scheduler.so python3 test.py
```

### 2. 确认拦截是否生效

```bash
# 看到 "intercepted" 日志说明拦截成功
ORION_LOG_LEVEL=3 LD_PRELOAD=./build/libgpu_scheduler.so python3 -c "
import torch
A = torch.randn(100, 100, device='cuda')
B = torch.randn(100, 100, device='cuda')
C = torch.mm(A, B)
"
```

### 3. Passthrough vs 调度模式

```python
import ctypes

lib = ctypes.CDLL("./build/libgpu_scheduler.so")

# Passthrough 模式 (不调度)
torch.mm(A, B)  # 直接执行，不经过调度器

# 调度模式
lib.orion_start_scheduler(2)
lib.orion_set_client_idx(0)
torch.mm(A, B)  # 经过调度器
lib.orion_stop_scheduler()
```

## 六、核心概念总结

| 概念 | 说明 |
|------|------|
| **LD_PRELOAD** | 让我们的同名函数先被调用 |
| **OperationRecord** | 记录一次 CUDA 操作的所有信息 |
| **ClientQueue** | 每个客户端一个操作队列 |
| **Scheduler Thread** | 从队列取操作并执行 |
| **HP/BE** | 高优先级/尽力而为客户端 |
| **Stream** | CUDA 的异步执行通道 |
| **wait_operation** | 客户端阻塞等待执行完成 |
| **mark_completed** | 调度器通知客户端可以继续 |
