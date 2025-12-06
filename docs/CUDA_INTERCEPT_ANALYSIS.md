# CUDA API 拦截机制分析

## 目标
理解 LD_PRELOAD 拦截 CUDA API 的具体做法，以及如何将调用改写为"记录 + 调度"。

---

## 1. 获取真实 CUDA 函数指针

### 1.1 函数指针类型定义

```cpp
// 定义每个 CUDA 函数的类型
using cudaMalloc_t = cudaError_t (*)(void**, size_t);
using cudaFree_t = cudaError_t (*)(void*);
using cudaLaunchKernel_t = cudaError_t (*)(const void*, dim3, dim3, void**, size_t, cudaStream_t);
// ...
```

### 1.2 全局存储结构

```cpp
static struct {
    cudaMalloc_t cudaMalloc;
    cudaFree_t cudaFree;
    cudaLaunchKernel_t cudaLaunchKernel;
    // ...
    bool initialized;
    std::mutex init_mutex;
} g_real_funcs = {nullptr};
```

### 1.3 获取函数指针的方式

```cpp
static void* get_cuda_func(const char* name) {
    // 方式 1: 尝试从已加载的库中获取
    if (!g_cudart_handle) {
        const char* lib_paths[] = {
            "libcudart.so.12",
            "libcudart.so.11",
            "libcudart.so",
            nullptr
        };
        
        for (int i = 0; lib_paths[i]; i++) {
            // RTLD_NOLOAD: 只在已加载的库中查找，不加载新库
            g_cudart_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_NOLOAD);
            if (g_cudart_handle) break;
        }
        
        // 方式 2: 如果没找到，加载库
        if (!g_cudart_handle) {
            for (int i = 0; lib_paths[i]; i++) {
                g_cudart_handle = dlopen(lib_paths[i], RTLD_NOW | RTLD_GLOBAL);
                if (g_cudart_handle) break;
            }
        }
    }
    
    // 方式 3: 从库句柄中获取符号
    if (g_cudart_handle) {
        void* fn = dlsym(g_cudart_handle, name);
        if (fn) return fn;
    }
    
    // 方式 4: 从全局符号表查找
    return dlsym(RTLD_DEFAULT, name);
}
```

### 1.4 初始化所有函数指针

```cpp
static void init_real_functions() {
    std::lock_guard<std::mutex> lock(g_real_funcs.init_mutex);
    if (g_real_funcs.initialized) return;
    
    g_real_funcs.cudaMalloc = (cudaMalloc_t)get_cuda_func("cudaMalloc");
    g_real_funcs.cudaFree = (cudaFree_t)get_cuda_func("cudaFree");
    g_real_funcs.cudaLaunchKernel = (cudaLaunchKernel_t)get_cuda_func("cudaLaunchKernel");
    // ...
    
    g_real_funcs.initialized = true;
}
```

---

## 2. LD_PRELOAD 拦截原理

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LD_PRELOAD 工作原理                          │
│                                                                     │
│   正常调用:                                                         │
│   App → cudaMalloc() → libcudart.so → GPU                          │
│                                                                     │
│   LD_PRELOAD 后:                                                    │
│   App → cudaMalloc() → libgpu_scheduler.so (我们的库)              │
│                             │                                       │
│                             ├─ 记录操作                             │
│                             ├─ 入队                                 │
│                             ├─ 等待                                 │
│                             │                                       │
│                             └─ 调度器 → g_real_funcs.cudaMalloc()  │
│                                              │                      │
│                                              └─ libcudart.so → GPU │
└─────────────────────────────────────────────────────────────────────┘
```

**关键**：我们定义了同名函数 `cudaMalloc`，链接器优先使用我们的版本。

---

## 3. 典型 API 拦截流程分析

### 3.1 cudaMalloc - 内存分配

```cpp
extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size) {
    // ========== 阶段 1: 透传检查 ==========
    SAFE_PASSTHROUGH(cudaMalloc, devPtr, size);
    // 展开后:
    // if (!initialized || tl_in_scheduler_execution) {
    //     return g_real_funcs.cudaMalloc(devPtr, size);  // 直接调用真实函数
    // }
    
    if (!is_capture_enabled()) {
        return real_cudaMalloc(devPtr, size);
    }
    
    int client_idx = get_current_client_idx();
    if (client_idx < 0) {
        return real_cudaMalloc(devPtr, size);  // 非管理线程，直接调用
    }
    
    // ========== 阶段 2: 创建操作记录 ==========
    auto op = create_operation(client_idx, OperationType::MALLOC);
    if (!op) return real_cudaMalloc(devPtr, size);
    
    // ========== 阶段 3: 设置参数 ==========
    op->params = MallocParams{devPtr, size};
    
    // ========== 阶段 4: 入队 ==========
    enqueue_operation(op);
    
    // ========== 阶段 5: 等待完成 ==========
    wait_operation(op);
    
    // ========== 阶段 6: 返回结果 ==========
    return op->result;
}
```

### 3.2 cudaLaunchKernel - 核心拦截点

```cpp
extern "C" cudaError_t cudaLaunchKernel(
    const void* func,      // kernel 函数指针
    dim3 gridDim,          // grid 配置
    dim3 blockDim,         // block 配置
    void** args,           // 参数数组指针
    size_t sharedMem,      // 共享内存大小
    cudaStream_t stream    // CUDA stream
) {
    // 透传检查...
    
    // NVTX 标记（用于 profiler 可视化）
    nvtxRangePush("Client:submit_kernel");
    
    // 创建操作记录
    auto op = create_operation(client_idx, OperationType::KERNEL_LAUNCH);
    
    // 设置 kernel 参数
    KernelLaunchParams kp;
    kp.func = func;
    kp.gridDim = gridDim;
    kp.blockDim = blockDim;
    kp.sharedMem = sharedMem;
    kp.stream = stream;
    kp.original_args = args;    // 保存原始参数指针
    kp.use_deep_copy = false;   // 当前不做深拷贝
    op->params = std::move(kp);
    
    // 入队
    enqueue_operation(op);
    
    // 等待调度器执行
    nvtxRangePush("Client:wait_scheduler");
    wait_operation(op);
    nvtxRangePop();
    
    nvtxRangePop();
    return op->result;
}
```

### 3.3 cudaMemcpy - 内存拷贝

```cpp
extern "C" cudaError_t cudaMemcpy(void* dst, const void* src, 
                                   size_t count, cudaMemcpyKind kind) {
    // 透传检查...
    
    auto op = create_operation(client_idx, OperationType::MEMCPY);
    
    // 设置参数
    op->params = MemcpyParams{
        dst,           // 目标地址
        src,           // 源地址
        count,         // 字节数
        kind,          // 拷贝类型 (H2D, D2H, D2D)
        nullptr,       // stream (同步版本为 nullptr)
        false          // is_async = false
    };
    
    enqueue_operation(op);
    wait_operation(op);
    return op->result;
}
```

---

## 4. 调度器执行真实操作

当调度器从队列取出操作后，调用 `execute_cuda_operation()`:

```cpp
cudaError_t execute_cuda_operation(OperationPtr op, cudaStream_t scheduler_stream) {
    // 设置重入标志，防止递归拦截
    tl_in_scheduler_execution = true;
    
    cudaError_t result;
    switch (op->type) {
        case OperationType::MALLOC:
            result = execute_malloc(op);
            break;
        case OperationType::KERNEL_LAUNCH:
            result = execute_kernel_launch(op, scheduler_stream);
            break;
        // ...
    }
    
    tl_in_scheduler_execution = false;
    return result;
}

// 执行真实的 malloc
cudaError_t execute_malloc(OperationPtr op) {
    auto& p = std::get<MallocParams>(op->params);
    cudaError_t err = g_real_funcs.cudaMalloc(p.devPtr, p.size);
    op->result_ptr = *p.devPtr;
    return err;
}

// 执行真实的 kernel launch
cudaError_t execute_kernel_launch(OperationPtr op, cudaStream_t scheduler_stream) {
    auto& p = std::get<KernelLaunchParams>(op->params);
    
    // 使用调度器的 stream，而不是客户端原来的 stream
    cudaStream_t stream_to_use = scheduler_stream ? scheduler_stream : p.stream;
    
    return g_real_funcs.cudaLaunchKernel(
        p.func, p.gridDim, p.blockDim,
        p.get_args(),
        p.sharedMem, stream_to_use
    );
}
```

---

## 5. 重入保护机制

```cpp
thread_local bool tl_in_scheduler_execution = false;

// SAFE_PASSTHROUGH 宏
#define SAFE_PASSTHROUGH(func_name, ...) \
    do { \
        if (!g_capture_state.initialized.load() || tl_in_scheduler_execution) { \
            return g_real_funcs.func_name(__VA_ARGS__); \
        } \
    } while(0)
```

**为什么需要？**

```
没有重入保护:
  调度器执行 cudaMalloc()
      → 被拦截
      → 创建操作记录
      → 入队
      → 等待自己执行 ← 死锁！

有重入保护:
  调度器执行 cudaMalloc()
      → tl_in_scheduler_execution = true
      → 被拦截
      → SAFE_PASSTHROUGH 检测到标志
      → 直接调用真实函数 ← 正确！
```

---

## 6. 直接转发 vs 记录+调度

### 6.1 直接转发的情况

| 条件 | 原因 |
|------|------|
| `!initialized` | 调度器未初始化 |
| `tl_in_scheduler_execution` | 调度器线程执行中，避免递归 |
| `!is_capture_enabled()` | 拦截被禁用 |
| `client_idx < 0` | 非管理线程（未调用 `set_client_idx`） |

```cpp
// 直接转发的代码路径
if (!is_capture_enabled()) {
    return real_cudaMalloc(devPtr, size);  // 直接调用，不经过队列
}
```

### 6.2 记录+调度的情况

| 条件 | 说明 |
|------|------|
| 调度器已初始化 | `initialized = true` |
| 拦截已启用 | `enabled = true` |
| 是管理线程 | `client_idx >= 0` |
| 不在调度器执行中 | `tl_in_scheduler_execution = false` |

```cpp
// 记录+调度的代码路径
auto op = create_operation(client_idx, TYPE);
op->params = {...};
enqueue_operation(op);    // 放入队列
wait_operation(op);       // 等待调度器执行
return op->result;
```

---

## 7. 总结图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       cudaMalloc() 被调用                               │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │ SAFE_PASSTHROUGH 检查          │
                    │ - initialized?                │
                    │ - tl_in_scheduler_execution?  │
                    └───────────────┬───────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                   YES                             NO
                    │                               │
                    ▼                               ▼
            ┌───────────────┐           ┌───────────────────────┐
            │ 直接转发       │           │ is_capture_enabled()? │
            │ real_cudaMalloc│           └───────────┬───────────┘
            └───────────────┘                       │
                                        ┌───────────┴───────────┐
                                       NO                      YES
                                        │                       │
                                        ▼                       ▼
                                ┌───────────────┐   ┌───────────────────────┐
                                │ 直接转发       │   │ client_idx >= 0?      │
                                └───────────────┘   └───────────┬───────────┘
                                                                │
                                                    ┌───────────┴───────────┐
                                                   NO                      YES
                                                    │                       │
                                                    ▼                       ▼
                                            ┌───────────────┐   ┌───────────────────┐
                                            │ 直接转发       │   │ 记录 + 调度        │
                                            └───────────────┘   │ 1. create_operation│
                                                                │ 2. 设置 params     │
                                                                │ 3. enqueue         │
                                                                │ 4. wait            │
                                                                │ 5. return result   │
                                                                └───────────────────┘
```

---

## 8. 拦截的 API 列表

| API | 操作类型 | 说明 |
|-----|---------|------|
| `cudaMalloc` | MALLOC | 设备内存分配 |
| `cudaFree` | FREE | 设备内存释放 |
| `cudaMemcpy` | MEMCPY | 同步内存拷贝 |
| `cudaMemcpyAsync` | MEMCPY_ASYNC | 异步内存拷贝 |
| `cudaMemset` | MEMSET | 同步内存设置 |
| `cudaMemsetAsync` | MEMSET_ASYNC | 异步内存设置 |
| `cudaLaunchKernel` | KERNEL_LAUNCH | **核心：所有 kernel 都经过这里** |
| `cudaDeviceSynchronize` | DEVICE_SYNC | 设备同步 |
| `cudaStreamSynchronize` | STREAM_SYNC | Stream 同步 |
