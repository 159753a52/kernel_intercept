# Python ctypes C 接口分析

## 目标
理解供 Python 侧调用的 C 接口，以及它们内部的初始化和清理工作。

---

## 1. 所有 extern "C" 接口一览

### 1.1 调度器接口 (scheduler.cpp)

```cpp
extern "C" {
    int orion_start_scheduler(int num_clients);  // 启动调度器
    void orion_stop_scheduler();                  // 停止调度器
    void orion_set_hp_latency(float latency_ms); // 设置 HP 延迟（未完整实现）
}
```

### 1.2 拦截层接口 (gpu_capture.cpp)

```cpp
extern "C" {
    int orion_init(int num_clients);        // 初始化拦截层
    void orion_shutdown();                   // 关闭拦截层
    void orion_set_client_idx(int idx);     // 设置当前线程的客户端索引
    int orion_get_client_idx();             // 获取当前线程的客户端索引
    void orion_set_enabled(int enabled);    // 启用/禁用拦截
    int orion_is_enabled();                 // 检查拦截是否启用
    void block(int phase);                  // 阻塞等待（预留）
}
```

### 1.3 Profiling 接口 (kernel_profile.cpp)

```cpp
extern "C" {
    int orion_load_profile(const char* filepath);      // 加载 kernel profile
    void orion_start_profiling(const char* model_name); // 开始 profiling
    void orion_end_profiling(const char* output_path);  // 结束 profiling
}
```

---

## 2. 核心接口实现详解

### 2.1 orion_start_scheduler

```cpp
// scheduler.cpp
extern "C" int orion_start_scheduler(int num_clients) {
    orion::SchedulerConfig config;
    return orion::start_scheduler(num_clients, config) ? 0 : -1;
}

// 内部实现
bool start_scheduler(int num_clients, const SchedulerConfig& config) {
    // 步骤 1: 初始化拦截层
    if (init_capture_layer(num_clients) != 0) {
        return false;
    }
    
    // 步骤 2: 初始化调度器
    if (!g_scheduler.init(num_clients, config)) {
        return false;
    }
    
    // 步骤 3: 启动调度器线程
    g_scheduler.start();
    return true;
}
```

### 2.2 orion_stop_scheduler

```cpp
extern "C" void orion_stop_scheduler() {
    orion::stop_scheduler();
}

void stop_scheduler() {
    g_scheduler.stop();           // 设置 running_ = false
    g_scheduler.join();           // 等待所有线程结束
    shutdown_capture_layer();     // 关闭拦截层
}
```

### 2.3 orion_set_client_idx

```cpp
extern "C" void orion_set_client_idx(int idx) {
    orion::set_current_client_idx(idx);
}

void set_current_client_idx(int idx) {
    tl_client_idx = idx;  // 设置 thread_local 变量
}
```

---

## 3. 初始化流程详解

### 3.1 init_capture_layer 做了什么

```cpp
int init_capture_layer(int num_clients) {
    // 1. 检查是否已初始化
    if (g_capture_state.initialized.load()) {
        return 0;
    }
    
    // 2. 参数检查
    if (num_clients <= 0 || num_clients > MAX_CLIENTS) {
        return -1;
    }
    
    // 3. 初始化日志
    init_log_level();
    
    // 4. 创建 per-client 队列
    g_capture_state.client_queues.resize(num_clients);
    for (int i = 0; i < num_clients; i++) {
        g_capture_state.client_queues[i] = std::make_unique<ClientQueue>();
    }
    
    // 5. 创建同步原语
    g_capture_state.client_blocked = new std::atomic<bool>[num_clients];
    g_capture_state.client_mutexes = new std::mutex[num_clients];
    g_capture_state.client_cvs = new std::condition_variable[num_clients];
    
    // 6. 设置状态标志
    g_capture_state.initialized.store(true);
    g_capture_state.enabled.store(true);
    
    return 0;
}
```

### 3.2 Scheduler::init 做了什么

```cpp
bool Scheduler::init(int num_clients, const SchedulerConfig& config) {
    // 1. 获取 GPU 信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, config.device_id);
    config_.num_sms = prop.multiProcessorCount;
    
    // 2. 创建 CUDA streams
    create_streams();
    //   - hp_stream_ (高优先级)
    //   - be_streams_[0..n-2] (低优先级)
    
    // 3. 预分配空间
    outstanding_kernels_.reserve(64);
    threads_.reserve(num_clients);
    
    // 4. 设置已初始化
    initialized_.store(true);
    
    return true;
}
```

### 3.3 Scheduler::start 做了什么

```cpp
void Scheduler::start() {
    running_.store(true);
    
    // 为每个客户端启动一个调度器线程
    for (int i = 0; i < num_clients_; i++) {
        threads_.emplace_back(&Scheduler::run_client, this, i);
    }
}
```

---

## 4. 完整初始化流程图

```
Python: lib.orion_start_scheduler(2)
                │
                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    orion_start_scheduler(2)                           │
│                              │                                        │
│                              ▼                                        │
│                    start_scheduler(2, config)                         │
│                              │                                        │
│              ┌───────────────┴───────────────┐                        │
│              │                               │                        │
│              ▼                               ▼                        │
│    init_capture_layer(2)            g_scheduler.init(2)               │
│              │                               │                        │
│              ├─ client_queues[0]             ├─ 获取 GPU SM 数        │
│              ├─ client_queues[1]             ├─ hp_stream_            │
│              ├─ client_blocked[0..1]         ├─ be_streams_[0]        │
│              ├─ client_mutexes[0..1]         └─ initialized = true    │
│              ├─ client_cvs[0..1]                                      │
│              ├─ initialized = true                                    │
│              └─ enabled = true                                        │
│                                                                       │
│                              │                                        │
│                              ▼                                        │
│                    g_scheduler.start()                                │
│                              │                                        │
│              ┌───────────────┴───────────────┐                        │
│              │                               │                        │
│              ▼                               ▼                        │
│     Thread 0 (HP)                   Thread 1 (BE)                     │
│     run_client(0)                   run_client(1)                     │
│         │                               │                             │
│         │ hp_stream_                    │ be_streams_[0]              │
│         │                               │                             │
│         ▼                               ▼                             │
│     while(running_) {               while(running_) {                 │
│       op = queue[0].try_pop()         op = queue[1].try_pop()         │
│       execute(op, hp_stream)          if (schedule_be()) {            │
│       mark_completed()                  execute(op, be_stream)        │
│     }                                   mark_completed()              │
│                                       }                               │
│                                     }                                 │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 5. Python 使用示例

```python
import ctypes
import threading

# 1. 加载库
lib = ctypes.CDLL("./build/libgpu_scheduler.so")

# 2. 设置函数签名
lib.orion_start_scheduler.argtypes = [ctypes.c_int]
lib.orion_start_scheduler.restype = ctypes.c_int
lib.orion_stop_scheduler.restype = None
lib.orion_set_client_idx.argtypes = [ctypes.c_int]

# 3. 启动调度器（2个客户端）
ret = lib.orion_start_scheduler(2)
if ret != 0:
    raise RuntimeError("Failed to start scheduler")

# 4. 在线程中设置客户端索引
def hp_worker():
    lib.orion_set_client_idx(0)  # HP 客户端
    # ... 执行 GPU 任务 ...

def be_worker():
    lib.orion_set_client_idx(1)  # BE 客户端
    # ... 执行 GPU 任务 ...

# 5. 启动工作线程
t1 = threading.Thread(target=hp_worker)
t2 = threading.Thread(target=be_worker)
t1.start(); t2.start()
t1.join(); t2.join()

# 6. 停止调度器
lib.orion_stop_scheduler()
```

---

## 6. 关闭流程

```
Python: lib.orion_stop_scheduler()
                │
                ▼
┌───────────────────────────────────────────────────────────────────────┐
│                    orion_stop_scheduler()                             │
│                              │                                        │
│                              ▼                                        │
│                       stop_scheduler()                                │
│                              │                                        │
│              ┌───────────────┼───────────────┐                        │
│              │               │               │                        │
│              ▼               ▼               ▼                        │
│     g_scheduler.stop()  g_scheduler.join()  shutdown_capture_layer() │
│              │               │               │                        │
│              │               │               │                        │
│     running_ = false    等待所有线程结束    shutdown = true           │
│     notify_all()                            notify_all()              │
│                                             destroy streams           │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 7. 接口调用顺序

```
┌─────────────────────────────────────────────────────────────────┐
│                       典型调用顺序                               │
│                                                                 │
│  1. orion_start_scheduler(num_clients)   // 初始化并启动        │
│         │                                                       │
│         ▼                                                       │
│  2. [各线程] orion_set_client_idx(idx)   // 设置身份            │
│         │                                                       │
│         ▼                                                       │
│  3. [各线程] 执行 PyTorch/CUDA 操作       // 被自动拦截调度      │
│         │                                                       │
│         ▼                                                       │
│  4. orion_stop_scheduler()               // 关闭                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 总结表

| 接口 | 文件 | 作用 | extern "C" |
|------|------|------|------------|
| `orion_start_scheduler` | scheduler.cpp | 初始化+启动调度器 | ✅ |
| `orion_stop_scheduler` | scheduler.cpp | 停止+清理调度器 | ✅ |
| `orion_set_client_idx` | gpu_capture.cpp | 设置线程的客户端索引 | ✅ |
| `orion_get_client_idx` | gpu_capture.cpp | 获取线程的客户端索引 | ✅ |
| `orion_set_enabled` | gpu_capture.cpp | 启用/禁用拦截 | ✅ |
| `orion_init` | gpu_capture.cpp | 单独初始化拦截层 | ✅ |
| `orion_shutdown` | gpu_capture.cpp | 单独关闭拦截层 | ✅ |
| `orion_load_profile` | kernel_profile.cpp | 加载 kernel profile | ✅ |

所有接口都是 `extern "C"` 导出，确保 Python ctypes 可以正确链接。
