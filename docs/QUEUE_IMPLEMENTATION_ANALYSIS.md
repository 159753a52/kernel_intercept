# GPU 操作捕获队列实现分析

## 目标
理解队列在 C++ 中的具体实现，包括锁、条件变量和生命周期管理。

---

## 1. 头文件 vs 实现文件对应关系

```
gpu_capture.h                          gpu_capture.cpp
─────────────────────────────────────────────────────────────
ClientQueue 类 (完整实现在 .h 中)  ←→  无（头文件中已内联）
CaptureLayerState 结构体声明      ←→  g_capture_state 全局变量定义
函数声明                          ←→  函数实现
```

**注意**：`ClientQueue` 是在头文件中完整实现的（内联类），因为方法都很短。

---

## 2. ClientQueue 队列实现详解

### 2.1 数据成员

```cpp
class ClientQueue {
private:
    std::queue<OperationPtr> queue_;  // 底层 STL 队列
    std::mutex mutex_;                 // 保护 queue_ 的互斥锁
    std::condition_variable cv_;       // 用于等待/通知
    std::atomic<bool> shutdown_;       // 退出标志
};
```

### 2.2 push() - 入队操作

```cpp
void push(OperationPtr op) {
    {
        std::lock_guard<std::mutex> lock(mutex_);  // 自动加锁
        queue_.push(std::move(op));                // 移动入队，避免拷贝
    }  // lock_guard 析构，自动解锁
    
    cv_.notify_one();  // 通知一个等待的消费者
}
```

**流程图**：
```
push(op)
    │
    ▼
┌─────────────────┐
│ lock(mutex_)    │  ← 获取锁
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ queue_.push(op) │  ← 放入队列
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ unlock(mutex_)  │  ← 释放锁（lock_guard 析构）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ cv_.notify_one()│  ← 唤醒一个等待者
└─────────────────┘
```

### 2.3 try_pop() - 非阻塞出队

```cpp
OperationPtr try_pop() {
    std::lock_guard<std::mutex> lock(mutex_);  // 加锁
    
    if (queue_.empty()) 
        return nullptr;   // 队列空，立即返回 nullptr
    
    OperationPtr op = std::move(queue_.front());  // 取出
    queue_.pop();                                  // 移除
    return op;
}
```

**流程图**：
```
try_pop()
    │
    ▼
┌─────────────────┐
│ lock(mutex_)    │
└────────┬────────┘
         │
         ▼
    ┌────────────┐
    │queue empty?│
    └────┬───────┘
         │
    ┌────┴────┐
    │         │
   YES       NO
    │         │
    ▼         ▼
┌───────┐  ┌──────────────┐
│return │  │op = front()  │
│nullptr│  │queue_.pop()  │
└───────┘  │return op     │
           └──────────────┘
```

### 2.4 wait_pop() - 阻塞出队

```cpp
OperationPtr wait_pop() {
    std::unique_lock<std::mutex> lock(mutex_);  // 用 unique_lock 因为 wait 需要
    
    // 条件等待：队列非空 或 收到 shutdown 信号
    cv_.wait(lock, [this] { 
        return !queue_.empty() || shutdown_; 
    });
    
    // 检查是否因为 shutdown 而退出
    if (shutdown_ && queue_.empty()) 
        return nullptr;
    
    OperationPtr op = std::move(queue_.front());
    queue_.pop();
    return op;
}
```

**流程图**：
```
wait_pop()
    │
    ▼
┌──────────────────────┐
│ lock(mutex_)         │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ while (queue_.empty() && !shutdown_) │  ← 条件不满足时循环
│ {                                    │
│     cv_.wait(lock);  ← 释放锁并睡眠  │
│                      ← 被唤醒后重新获取锁
│ }                                    │
└──────────────────┬───────────────────┘
                   │
                   ▼
          ┌────────────────┐
          │ shutdown_ &&   │
          │ queue_.empty()? │
          └───────┬────────┘
                  │
          ┌───────┴───────┐
         YES             NO
          │               │
          ▼               ▼
    ┌──────────┐   ┌─────────────┐
    │ return   │   │ op=front()  │
    │ nullptr  │   │ pop()       │
    └──────────┘   │ return op   │
                   └─────────────┘
```

### 2.5 shutdown() - 关闭队列

```cpp
void shutdown() {
    shutdown_ = true;      // 设置退出标志
    cv_.notify_all();      // 唤醒所有等待者
}
```

---

## 3. 同步原语详解

### 3.1 std::mutex - 互斥锁

```cpp
std::mutex mutex_;

// 使用方式1：手动加锁解锁（容易忘记解锁）
mutex_.lock();
// 临界区代码
mutex_.unlock();

// 使用方式2：lock_guard（RAII，推荐）
{
    std::lock_guard<std::mutex> lock(mutex_);
    // 临界区代码
}  // 自动解锁

// 使用方式3：unique_lock（更灵活，可以手动解锁）
std::unique_lock<std::mutex> lock(mutex_);
// 可以手动 lock.unlock() 和 lock.lock()
```

### 3.2 std::condition_variable - 条件变量

```cpp
std::condition_variable cv_;

// 等待（必须配合 unique_lock 使用）
std::unique_lock<std::mutex> lock(mutex_);
cv_.wait(lock, [&]{ return condition; });
// wait 做了什么：
//   1. 检查 condition，如果为 true，直接返回
//   2. 如果为 false，释放 lock，线程进入睡眠
//   3. 被 notify 唤醒后，重新获取 lock
//   4. 再次检查 condition，重复上述过程

// 通知
cv_.notify_one();   // 唤醒一个等待者
cv_.notify_all();   // 唤醒所有等待者
```

### 3.3 std::atomic - 原子变量

```cpp
std::atomic<bool> shutdown_{false};

shutdown_.store(true);   // 原子写入
bool val = shutdown_.load();  // 原子读取
shutdown_ = true;  // 也是原子操作（语法糖）
```

---

## 4. 生命周期管理

### 4.1 初始化流程

```cpp
int init_capture_layer(int num_clients) {
    // 1. 检查是否已初始化（原子操作）
    if (g_capture_state.initialized.load()) {
        return 0;  // 已初始化，直接返回
    }
    
    // 2. 参数检查
    if (num_clients <= 0 || num_clients > MAX_CLIENTS) {
        return -1;
    }
    
    // 3. 创建 per-client 队列
    g_capture_state.client_queues.resize(num_clients);
    for (int i = 0; i < num_clients; i++) {
        g_capture_state.client_queues[i] = std::make_unique<ClientQueue>();
    }
    
    // 4. 创建同步原语数组
    g_capture_state.client_blocked = new std::atomic<bool>[num_clients];
    g_capture_state.client_mutexes = new std::mutex[num_clients];
    g_capture_state.client_cvs = new std::condition_variable[num_clients];
    
    // 5. 设置状态标志
    g_capture_state.shutdown.store(false);
    g_capture_state.initialized.store(true);
    g_capture_state.enabled.store(true);
    
    return 0;
}
```

### 4.2 关闭流程

```cpp
void shutdown_capture_layer() {
    // 1. 检查是否已初始化
    if (!g_capture_state.initialized.load()) {
        return;
    }
    
    // 2. 设置关闭标志
    g_capture_state.shutdown.store(true);
    g_capture_state.enabled.store(false);
    
    // 3. 唤醒所有等待的 client
    for (int i = 0; i < g_capture_state.num_clients; i++) {
        g_capture_state.client_blocked[i].store(false);
        g_capture_state.client_cvs[i].notify_all();
        g_capture_state.client_queues[i]->shutdown();  // 通知队列关闭
    }
    
    // 4. 唤醒调度器
    g_capture_state.scheduler_cv.notify_all();
    
    // 5. 设置未初始化状态
    g_capture_state.initialized.store(false);
}
```

### 4.3 内存管理

```cpp
// 初始化时：动态分配
g_capture_state.client_blocked = new std::atomic<bool>[num_clients];
g_capture_state.client_mutexes = new std::mutex[num_clients];
g_capture_state.client_cvs = new std::condition_variable[num_clients];

// 析构时：自动释放（在 CaptureLayerState 析构函数中）
~CaptureLayerState() {
    delete[] client_blocked;
    delete[] client_mutexes;
    delete[] client_cvs;
}
```

---

## 5. 边界条件处理

### 5.1 队列空

| 方法 | 处理方式 |
|------|---------|
| `try_pop()` | 返回 `nullptr`，调用者检查 |
| `wait_pop()` | 阻塞等待，直到有元素 |
| `peek()` | 返回 `nullptr` |

### 5.2 队列满

**本实现没有队列大小限制**（使用 `std::queue`，动态增长）。

如果需要限制大小，可以修改：
```cpp
void push(OperationPtr op) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    // 等待队列有空间
    cv_not_full_.wait(lock, [this] { 
        return queue_.size() < MAX_SIZE || shutdown_; 
    });
    
    queue_.push(std::move(op));
    cv_.notify_one();
}
```

### 5.3 退出信号

```cpp
void shutdown() {
    shutdown_ = true;   // 设置标志
    cv_.notify_all();   // 唤醒所有等待者
}

// wait_pop 中的处理
cv_.wait(lock, [this] { 
    return !queue_.empty() || shutdown_;  // 收到 shutdown 也返回
});

if (shutdown_ && queue_.empty()) 
    return nullptr;  // shutdown 且队列空，返回 nullptr 告诉调用者退出
```

---

## 6. 生产者-消费者模型

### 6.1 角色对应

```
生产者 = 客户端线程（调用 CUDA API）
消费者 = 调度器线程（执行真实 CUDA 函数）
队列   = ClientQueue
产品   = OperationRecord
```

### 6.2 伪代码描述

```
// ========== 生产者（客户端线程）==========
function producer(client_idx):
    while (有工作要做):
        // 调用被拦截的 CUDA 函数
        result = cudaMalloc(&ptr, size)  // 被拦截
        
        // 拦截函数内部：
        //   op = create_operation(MALLOC)
        //   op->params = {ptr, size}
        //   queue[client_idx].push(op)      ← 生产
        //   op->wait_completion()           ← 等待消费者处理完
        //   return op->result


// ========== 消费者（调度器线程）==========
function consumer(client_idx):
    while (!shutdown):
        // 从队列取出操作
        op = queue[client_idx].try_pop()   ← 消费
        
        if (op != nullptr):
            // 执行真实的 CUDA 操作
            result = execute_operation(op)
            
            // 通知生产者完成
            op->mark_completed(result)      ← 通知
        else:
            // 队列空，等待
            wait_for_new_operation()
```

### 6.3 时序图

```
生产者线程                     队列                      消费者线程
     │                          │                           │
     │  cudaMalloc() 被拦截     │                           │
     │                          │                           │
     │  create_operation()      │                           │
     │  设置 params             │                           │
     │                          │                           │
     │──── push(op) ───────────>│                           │
     │                          │                           │
     │                          │<──── try_pop() ───────────│
     │                          │                           │
     │                          │────── return op ─────────>│
     │                          │                           │
     │  wait_completion()       │      execute_operation()  │
     │       │                  │            │              │
     │       │ cv.wait()        │            │              │
     │       │ (阻塞)           │      real_cudaMalloc()    │
     │       │                  │            │              │
     │       │<─────────────────────── mark_completed() ────│
     │       │  cv.notify_all() │                           │
     │                          │                           │
     │  return result           │                           │
     │                          │                           │
```

---

## 7. 关键设计点总结

### 7.1 为什么用 shared_ptr？

```cpp
using OperationPtr = std::shared_ptr<OperationRecord>;
```

- 生产者和消费者共享同一个 `OperationRecord`
- 自动内存管理，无需手动 delete
- 引用计数为 0 时自动释放

### 7.2 为什么 OperationRecord 禁止拷贝和移动？

```cpp
OperationRecord(const OperationRecord&) = delete;
OperationRecord(OperationRecord&&) = delete;
```

- 包含 `std::mutex` 和 `std::condition_variable`
- 这些类型不可拷贝/移动
- 通过 `shared_ptr` 共享，不需要拷贝

### 7.3 为什么用 lock_guard vs unique_lock？

| 类型 | 特点 | 使用场景 |
|------|------|---------|
| `lock_guard` | 简单，只能加锁解锁一次 | 简单临界区 |
| `unique_lock` | 灵活，可以手动解锁/重锁 | 配合条件变量 |

```cpp
// lock_guard：简单场景
void push(...) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(op);
}

// unique_lock：需要条件等待
OperationPtr wait_pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, ...);  // wait 内部会临时释放锁
    ...
}
```

### 7.4 notify_one vs notify_all

| 方法 | 作用 | 使用场景 |
|------|------|---------|
| `notify_one` | 唤醒一个等待者 | 每次只有一个消费者能处理 |
| `notify_all` | 唤醒所有等待者 | shutdown 时确保所有都退出 |
