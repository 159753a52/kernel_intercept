# 调度器 API 分析

## 目标
弄清调度器暴露给外部（Python 和拦截层）的 C 接口，以及核心数据结构。

---

## 1. 对外可见的 C 接口

### 1.1 调度器接口 (scheduler.cpp)

| 函数 | 作用 | 参数 |
|------|------|------|
| `orion_start_scheduler(int num_clients)` | 启动调度器 | 客户端数量 |
| `orion_stop_scheduler()` | 停止调度器 | 无 |
| `orion_set_hp_latency(float latency_ms)` | 设置 HP 延迟（未完整实现） | 延迟毫秒 |

### 1.2 拦截层接口 (gpu_capture.cpp)

| 函数 | 作用 | 参数 |
|------|------|------|
| `orion_init(int num_clients)` | 初始化拦截层 | 客户端数量 |
| `orion_shutdown()` | 关闭拦截层 | 无 |
| `orion_set_client_idx(int idx)` | 设置当前线程的客户端索引 | 索引 |
| `orion_get_client_idx()` | 获取当前线程的客户端索引 | 无 |
| `orion_set_enabled(int enabled)` | 启用/禁用拦截 | 0或1 |
| `orion_is_enabled()` | 检查拦截是否启用 | 无 |

### 1.3 Profiling 接口 (kernel_profile.cpp)

| 函数 | 作用 | 参数 |
|------|------|------|
| `orion_load_profile(const char* filepath)` | 加载 kernel profile | 文件路径 |
| `orion_start_profiling(const char* model_name)` | 开始 profiling | 模型名 |
| `orion_end_profiling(const char* output_path)` | 结束并保存 profiling | 输出路径 |

---

## 2. 核心数据结构

### 2.1 SchedulerConfig - 调度配置

```cpp
struct SchedulerConfig {
    float sm_threshold_ratio = 0.5f;      // BE 最大 SM 占用比例
    float dur_threshold_ratio = 0.025f;   // BE 累计时间/HP延迟 比例
    float hp_request_latency_ms = 10.0f;  // HP 请求平均延迟
    int poll_interval_us = 10;            // 轮询间隔（微秒）
    bool interference_aware = true;       // 是否启用干扰感知
    int device_id = 0;                    // GPU 设备 ID
    int num_sms = 0;                      // SM 数量（运行时获取）
};
```

### 2.2 Scheduler - 调度器类

```cpp
class Scheduler {
private:
    // ========== 配置 ==========
    SchedulerConfig config_;
    
    // ========== 线程管理 ==========
    std::vector<std::thread> threads_;    // 每个客户端一个线程
    std::atomic<bool> running_;           // 运行标志
    std::atomic<bool> initialized_;       // 初始化标志
    
    // ========== CUDA Streams ==========
    cudaStream_t hp_stream_;              // 高优先级 stream (client 0)
    std::vector<cudaStream_t> be_streams_;// BE streams (client 1, 2, ...)
    
    // ========== 调度状态 ==========
    std::atomic<bool> hp_task_running_;   // HP 任务是否在执行
    OperationPtr current_hp_op_;          // 当前 HP 操作
    std::atomic<int> active_be_count_;    // 活跃 BE 操作数
    
    // ========== BE 调度控制 ==========
    std::mutex be_schedule_mutex_;
    std::condition_variable be_schedule_cv_;
    float cumulative_be_duration_ms_;     // 累计 BE 时间
    
    // ========== 统计 ==========
    Stats stats_;
    int num_clients_;
};
```

### 2.3 Stats - 统计信息

```cpp
struct Stats {
    uint64_t hp_ops_scheduled = 0;    // HP 操作调度次数
    uint64_t be_ops_scheduled = 0;    // BE 操作调度次数
    uint64_t be_ops_rejected = 0;     // BE 操作拒绝次数
    uint64_t total_wait_time_us = 0;  // 总等待时间
};
```

---

## 3. 客户端索引含义

```
Client 0 = HP (High Priority)    高优先级客户端
Client 1 = BE (Best Effort) #1   尽力服务客户端 1
Client 2 = BE (Best Effort) #2   尽力服务客户端 2
...
```

调度器为每个客户端创建独立的：
- 调度线程（执行该客户端的操作）
- CUDA Stream（HP 用高优先级，BE 用低优先级）

---

## 4. 调用顺序

### 4.1 Python 使用示例

```python
import ctypes
import threading
import torch

# ========== 阶段 1: 加载库 ==========
lib = ctypes.CDLL("./build/libgpu_scheduler.so")

# ========== 阶段 2: 启动调度器 ==========
# 参数: 客户端数量（1 HP + N BE）
lib.orion_start_scheduler(2)  # 2个客户端: client 0 (HP) + client 1 (BE)

# ========== 阶段 3: 执行任务 ==========
def hp_task():
    lib.orion_set_client_idx(0)  # 设置为 HP 客户端
    # PyTorch 操作会被拦截并通过调度器执行
    C = torch.mm(A, B)

def be_task():
    lib.orion_set_client_idx(1)  # 设置为 BE 客户端
    # 这些操作可能与 HP 并发执行
    D = torch.add(X, Y)

t1 = threading.Thread(target=hp_task)
t2 = threading.Thread(target=be_task)
t1.start(); t2.start()
t1.join(); t2.join()

# ========== 阶段 4: 停止调度器 ==========
lib.orion_stop_scheduler()
```

### 4.2 调用时序图

```
Python 主线程              HP 线程                 BE 线程                调度器
     │                       │                       │                      │
     │ orion_start_scheduler(2)                      │                      │
     │───────────────────────────────────────────────────────────────────-->│
     │                       │                       │    init()            │
     │                       │                       │    create_streams()  │
     │                       │                       │    start() ────────> │
     │                       │                       │         启动2个线程   │
     │                       │                       │                      │
     │  创建线程 t1, t2      │                       │                      │
     │───────>│              │                       │                      │
     │        │              │                       │                      │
     │        │ set_client_idx(0)                    │                      │
     │        │──────────────────────────────────────────────────────────-->│
     │        │              │                       │                      │
     │        │ torch.mm()   │                       │                      │
     │        │    │         │                       │                      │
     │        │    │ cudaLaunchKernel (被拦截)       │                      │
     │        │    │─────────────────────────────────────────────────────-->│
     │        │    │         │                       │  HP 队列入队          │
     │        │    │ wait... │                       │  执行操作             │
     │        │    │<────────────────────────────────────────────────────────│
     │        │    │         │                       │                      │
     │        │              │ set_client_idx(1)     │                      │
     │        │              │──────────────────────────────────────────────>│
     │        │              │                       │                      │
     │        │              │ torch.add()           │                      │
     │        │              │    │                  │                      │
     │        │              │    │──────────────────────────────────────-->│
     │        │              │    │                  │  BE 队列入队          │
     │        │              │    │ wait...          │  检查是否允许并发     │
     │        │              │    │<─────────────────────────────────────────│
     │        │              │                       │                      │
     │  t1.join(), t2.join() │                       │                      │
     │<───────│──────────────│                       │                      │
     │                       │                       │                      │
     │ orion_stop_scheduler()│                       │                      │
     │───────────────────────────────────────────────────────────────────-->│
     │                       │                       │    stop()            │
     │                       │                       │    join()            │
```

---

## 5. 伪代码：典型程序流程

```python
# ============================================================
# 典型使用流程
# ============================================================

# 1. 加载调度器库
lib = load_library("libgpu_scheduler.so")

# 2. 启动调度器
#    - 初始化拦截层（创建 per-client 队列）
#    - 初始化调度器（创建 CUDA streams）
#    - 启动调度线程（每个客户端一个）
lib.orion_start_scheduler(num_clients=2)

# 3. 在各线程中设置客户端索引
def worker(client_id):
    # 告诉拦截层"我是哪个客户端"
    lib.orion_set_client_idx(client_id)
    
    # 之后的所有 CUDA 调用都会被拦截
    # 并放入对应客户端的队列
    result = do_gpu_work()
    
    return result

# 4. 创建并运行线程
threads = []
for i in range(num_clients):
    t = Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

# 5. 等待所有线程完成
for t in threads:
    t.join()

# 6. 停止调度器
#    - 通知所有队列 shutdown
#    - 等待调度线程结束
#    - 销毁 CUDA streams
lib.orion_stop_scheduler()
```

---

## 6. 关键设计点

### 6.1 为什么用 `extern "C"`？

```cpp
extern "C" {
    int orion_start_scheduler(int num_clients);
}
```

- C++ 会对函数名进行 name mangling（如 `_Z21orion_start_scheduleri`）
- Python ctypes 需要原始函数名
- `extern "C"` 告诉编译器使用 C 链接约定，保持函数名不变

### 6.2 为什么每个客户端一个线程？

```
单线程调度器的问题：
  HP 操作执行时，BE 操作必须等待
  无法真正并发

多线程调度器：
  HP 线程 → 立即执行 HP 操作
  BE 线程 → 可以和 HP 并发（如果 schedule_be() 允许）
```

### 6.3 `orion_set_client_idx` 的作用

```cpp
thread_local int tl_client_idx = -1;  // 每个线程独立

void orion_set_client_idx(int idx) {
    tl_client_idx = idx;
}
```

- 使用 `thread_local` 存储客户端索引
- 每个线程调用一次，设置自己的身份
- 后续 CUDA 调用被拦截时，通过 `get_current_client_idx()` 知道放入哪个队列

---

## 7. 接口总结表

| 阶段 | 函数 | 说明 |
|------|------|------|
| 初始化 | `orion_start_scheduler(n)` | 启动调度器，n 个客户端 |
| 设置身份 | `orion_set_client_idx(i)` | 当前线程是第 i 个客户端 |
| 执行任务 | （自动拦截） | CUDA 调用自动进入调度 |
| 查询状态 | `orion_is_enabled()` | 检查拦截是否启用 |
| 关闭 | `orion_stop_scheduler()` | 停止调度器 |
