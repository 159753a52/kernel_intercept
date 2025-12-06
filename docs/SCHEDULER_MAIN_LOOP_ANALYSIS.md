# 调度器主循环分析

## 目标
理解调度器如何从队列取操作、判断并发、执行到 CUDA stream。

---

## 0. 主循环代码逐行注释

```cpp
void Scheduler::run_client(int client_idx) {
    // ========== 初始化阶段 ==========
    
    // 判断当前线程是 HP 还是 BE
    bool is_hp = (client_idx == 0);  // client 0 = HP
    
    // 选择对应的 CUDA stream
    cudaStream_t my_stream;
    if (is_hp) {
        my_stream = hp_stream_;           // 高优先级 stream
    } else {
        my_stream = be_streams_[client_idx - 1];  // 低优先级 stream
    }
    
    // ========== 主循环 ==========
    while (running_.load()) {  // 检查是否还在运行
        
        // -------- 步骤 1: 从队列取操作 --------
        OperationPtr op = client_queues[client_idx]->try_pop();  // 非阻塞
        
        if (op) {  // 取到了操作
            
            if (is_hp) {
                // -------- HP 路径：直接执行 --------
                
                hp_task_running_ = true;      // 标记 HP 在执行
                current_hp_op_ = op;          // 记录当前 HP 操作
                
                execute_operation(op, my_stream);  // 执行！
                op->mark_completed(err);           // 通知客户端完成
                
                hp_task_running_ = false;     // 标记 HP 执行完毕
                current_hp_op_ = nullptr;
                
                be_schedule_cv_.notify_all(); // 唤醒等待的 BE 线程
                
            } else {
                // -------- BE 路径：需要调度决策 --------
                
                bool allowed = false;
                {
                    unique_lock lock(be_schedule_mutex_);
                    
                    // 等待直到：允许执行 或 HP 空闲 或 shutdown
                    be_schedule_cv_.wait(lock, [&]() {
                        allowed = schedule_be(current_hp_op_, op);
                        return allowed || !hp_task_running_ || !running_;
                    });
                }
                
                if (allowed || !hp_task_running_) {
                    execute_operation(op, my_stream);  // 执行！
                    op->mark_completed(err);
                }
            }
            
        } else {  // 队列空
            // -------- 步骤 2: 等待新操作 --------
            unique_lock lock(scheduler_mutex);
            scheduler_cv.wait_for(lock, poll_interval, [&] {
                return !running_ || !queue[client_idx]->empty();
            });
        }
    }
    
    // ========== 清理阶段：处理剩余操作 ==========
    while (!queue[client_idx]->empty()) {
        auto op = queue[client_idx]->try_pop();
        execute_operation(op, my_stream);
        op->mark_completed();
    }
    cudaStreamSynchronize(my_stream);  // 等待所有操作完成
}
```

---

## 1. 调度器架构：多线程模型

```
┌─────────────────────────────────────────────────────────────────┐
│                         Scheduler                                │
│                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│   │ Thread 0    │    │ Thread 1    │    │ Thread 2    │   ...   │
│   │ (HP client) │    │ (BE client) │    │ (BE client) │         │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│          │                  │                  │                 │
│          ▼                  ▼                  ▼                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│   │ HP Queue    │    │ BE Queue 1  │    │ BE Queue 2  │         │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘         │
│          │                  │                  │                 │
│          ▼                  ▼                  ▼                 │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│   │ hp_stream_  │    │ be_stream_0 │    │ be_stream_1 │         │
│   │ (高优先级)   │    │ (低优先级)  │    │ (低优先级)   │         │
│   └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**关键设计**：每个客户端有独立的线程、队列和 CUDA stream。

---

## 2. 主循环：run_client()

```cpp
void Scheduler::run_client(int client_idx) {
    // 确定使用哪个 stream
    bool is_hp = (client_idx == 0);
    cudaStream_t my_stream = is_hp ? hp_stream_ : be_streams_[client_idx - 1];
    
    while (running_.load()) {
        // ========== 步骤 1: 尝试取操作 ==========
        OperationPtr op = client_queues[client_idx]->try_pop();
        
        if (op) {
            if (is_hp) {
                // ========== 步骤 2a: HP 操作 - 直接执行 ==========
                hp_task_running_ = true;
                execute_operation(op, my_stream);
                op->mark_completed();
                hp_task_running_ = false;
                
                // 唤醒等待的 BE 线程
                be_schedule_cv_.notify_all();
            } else {
                // ========== 步骤 2b: BE 操作 - 需要调度决策 ==========
                bool allowed = schedule_be(current_hp_op_, op);
                
                if (allowed) {
                    execute_operation(op, my_stream);
                    op->mark_completed();
                }
            }
        } else {
            // ========== 步骤 3: 队列空，等待 ==========
            scheduler_cv.wait_for(lock, poll_interval);
        }
    }
}
```

---

## 3. 主循环流程图

```
run_client(client_idx) 开始
         │
         ▼
    ┌─────────┐
    │running_?│
    └────┬────┘
         │
    ┌────┴────┐
   NO        YES
    │         │
    ▼         ▼
  退出    ┌──────────┐
          │try_pop() │
          └────┬─────┘
               │
          ┌────┴────┐
        nullptr    有 op
          │         │
          ▼         ▼
    ┌──────────┐  ┌─────────┐
    │等待新操作 │  │ is_hp?  │
    │cv.wait() │  └────┬────┘
    └────┬─────┘       │
         │        ┌────┴────┐
         │       YES       NO
         │        │         │
         │        ▼         ▼
         │  ┌──────────┐  ┌───────────────┐
         │  │直接执行   │  │检查 schedule_be│
         │  │hp_running │  │  是否允许并发? │
         │  │ = true   │  └───────┬───────┘
         │  └────┬─────┘          │
         │       │           ┌────┴────┐
         │       ▼          YES       NO
         │  ┌──────────┐     │         │
         │  │execute() │     ▼         ▼
         │  │on hp_strm│ ┌────────┐ ┌────────┐
         │  └────┬─────┘ │execute │ │等待或  │
         │       │       │on be_  │ │强制执行│
         │       ▼       │stream  │ └────┬───┘
         │  ┌──────────┐ └────┬───┘      │
         │  │hp_running│      │          │
         │  │ = false  │      ▼          │
         │  │notify_all│ mark_completed  │
         │  └────┬─────┘      │          │
         │       │            │          │
         └───────┴────────────┴──────────┘
                        │
                        ▼
                   回到 while 开头
```

---

## 4. HP/BE 调度策略：schedule_be()

```cpp
bool Scheduler::schedule_be(const OperationPtr& hp_op, const OperationPtr& be_op) {
    // ========== 检查 1: HP 任务是否在执行 ==========
    if (!hp_task_running_.load()) {
        return true;  // HP 空闲，BE 可以执行
    }
    
    // ========== 检查 2: 干扰感知是否启用 ==========
    if (!config_.interference_aware) {
        return true;  // 不启用干扰感知，依赖 stream 优先级
    }
    
    // ========== 检查 3: SM 需求 ==========
    int sm_needed = be_op->sm_needed;
    if (sm_needed >= config_.get_sm_threshold()) {
        return false;  // BE 需要太多 SM，会干扰 HP
    }
    
    // ========== 检查 4: Profile 类型是否互补 ==========
    if (!is_complementary(hp_op->profile_type, be_op->profile_type)) {
        return false;  // 不互补，会争抢资源
    }
    
    // ========== 检查 5: 累计 BE 时间 ==========
    if (cumulative_be_duration_ms_ + be_op->duration > threshold) {
        return false;  // BE 已经占用太多时间
    }
    
    cumulative_be_duration_ms_ += be_op->duration;
    return true;
}
```

### 调度决策流程图

```
schedule_be(hp_op, be_op)
         │
         ▼
    ┌─────────────────┐
    │ HP 任务在执行?   │
    └────────┬────────┘
             │
        ┌────┴────┐
       NO        YES
        │         │
        ▼         ▼
   return true  ┌─────────────────┐
                │ 干扰感知启用?    │
                └────────┬────────┘
                         │
                    ┌────┴────┐
                   NO        YES
                    │         │
                    ▼         ▼
               return true  ┌─────────────────┐
                            │ SM 需求 < 50%?  │
                            └────────┬────────┘
                                     │
                                ┌────┴────┐
                               NO        YES
                                │         │
                                ▼         ▼
                           return false ┌─────────────────┐
                                        │ Profile 互补?   │
                                        └────────┬────────┘
                                                 │
                                            ┌────┴────┐
                                           NO        YES
                                            │         │
                                            ▼         ▼
                                       return false ┌─────────────────┐
                                                    │ 累计时间 < 2.5%? │
                                                    └────────┬────────┘
                                                             │
                                                        ┌────┴────┐
                                                       NO        YES
                                                        │         │
                                                        ▼         ▼
                                                   return false  return true
```

---

## 5. Profile 互补性判断

```cpp
bool is_complementary(ProfileType hp_type, ProfileType be_type) {
    // UNKNOWN 类型认为可能互补
    if (hp_type == UNKNOWN || be_type == UNKNOWN) {
        return true;
    }
    
    // 一个 compute-bound，一个 memory-bound 时互补
    return (hp_type == COMPUTE_BOUND && be_type == MEMORY_BOUND) ||
           (hp_type == MEMORY_BOUND && be_type == COMPUTE_BOUND);
}
```

**为什么互补很重要？**

```
GPU 资源分为两类：
  - 计算单元 (SM, CUDA cores)
  - 内存带宽 (HBM bandwidth)

Compute-bound kernel: 主要用计算单元，内存带宽空闲
Memory-bound kernel:  主要用内存带宽，计算单元空闲

互补执行：
  HP (compute) + BE (memory) → 两种资源都被利用
  HP (memory)  + BE (compute) → 两种资源都被利用

不互补执行：
  HP (compute) + BE (compute) → 计算单元争抢，HP 被干扰！
```

---

## 6. HP/BE 时间轴交错执行示例

### 场景：HP 执行 GEMM（计算密集），BE 执行 Memcpy（内存密集）

```
时间 →

没有调度器（串行执行）：
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  HP Stream: [====GEMM====][====GEMM====][====GEMM====]            │
│                                                                   │
│  BE Stream:                                            [Memcpy]   │
│                                                                   │
│  GPU利用率:  计算100%      计算100%      计算100%       带宽100%   │
│             带宽 20%       带宽 20%       带宽 20%       计算 0%   │
│                                                                   │
│  总时间: ████████████████████████████████████████████████████████ │
└───────────────────────────────────────────────────────────────────┘

有调度器（并发执行）：
┌───────────────────────────────────────────────────────────────────┐
│                                                                   │
│  HP Stream: [====GEMM====][====GEMM====][====GEMM====]            │
│                    │            │            │                    │
│  BE Stream:    [Memcpy]     [Memcpy]     [Memcpy]                 │
│                                                                   │
│  GPU利用率:  计算100%      计算100%      计算100%                  │
│             带宽 80%       带宽 80%       带宽 80%                  │
│                                                                   │
│  总时间: ████████████████████████████████                          │
│                                                                   │
│  节省时间: ██████████████                                          │
└───────────────────────────────────────────────────────────────────┘
```

### 更详细的时间轴

```
时间    0    10   20   30   40   50   60   70   80   90   100
        │    │    │    │    │    │    │    │    │    │    │
HP:     [========GEMM 1========][========GEMM 2========]
        |    |    |    |    |    |    |    |    |    |
BE:     [Mc1][Mc2]     [Mc3][Mc4]          [Mc5][Mc6]
        ↑    ↑         ↑    ↑              ↑    ↑
        │    │         │    │              │    │
        └────┴─────────┴────┴──────────────┴────┴─── 在 HP 计算期间
                                                      BE 利用空闲内存带宽

调度决策时机：
  t=0:  schedule_be(GEMM1, Mc1) → HP=compute, BE=memory → 互补 → 允许
  t=10: schedule_be(GEMM1, Mc2) → 累计时间 OK → 允许
  t=20: schedule_be(GEMM1, Mc3) → 累计时间超限 → 拒绝，等 GEMM1 完成
  t=30: HP 完成，cumulative 重置，BE 继续...
```

---

## 7. 为什么这样的调度能提升利用率？

### 7.1 GPU 资源的两个维度

```
┌─────────────────────────────────────────────────────────────┐
│                        GPU 资源                              │
│                                                              │
│   ┌─────────────────────┐    ┌─────────────────────┐        │
│   │   计算资源 (SM)      │    │   内存带宽 (HBM)    │        │
│   │                     │    │                     │        │
│   │  - CUDA cores      │    │  - 读带宽           │        │
│   │  - Tensor cores    │    │  - 写带宽           │        │
│   │  - Register files  │    │  - L2 cache         │        │
│   └─────────────────────┘    └─────────────────────┘        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 单任务执行时的资源浪费

```
Compute-bound GEMM 执行时：
  计算资源: ████████████ 100%
  内存带宽: ████░░░░░░░░  30%  ← 浪费 70%

Memory-bound Memcpy 执行时：
  计算资源: ░░░░░░░░░░░░   0%  ← 浪费 100%
  内存带宽: ████████████ 100%
```

### 7.3 互补任务并发执行

```
GEMM + Memcpy 并发：
  计算资源: ████████████ 100% (GEMM)
  内存带宽: ████████████  30% (GEMM) + 70% (Memcpy) = 100%

结果：两种资源都充分利用！
```

### 7.4 数学解释

```
假设：
  HP GEMM 执行时间 = T_hp
  BE Memcpy 总时间 = T_be

串行执行：
  总时间 = T_hp + T_be

并发执行（BE 完全隐藏在 HP 内存空闲期间）：
  总时间 ≈ T_hp  （T_be 被隐藏）

吞吐量提升 = (T_hp + T_be) / T_hp - 1 = T_be / T_hp
```

---

## 8. 调度约束：防止干扰 HP

### 8.1 为什么需要约束？

如果 BE 任务太多或太重，会：
1. **抢占计算资源**：BE 占用 SM，HP 分到的 SM 减少
2. **抢占内存带宽**：BE 读写数据，HP 内存访问变慢
3. **增加尾延迟**：HP 完成时间不稳定

### 8.2 三道防线

```
┌──────────────────────────────────────────────────────────┐
│                     调度约束                              │
│                                                          │
│  防线 1: SM 阈值 (50%)                                   │
│    BE 的 SM 需求 < GPU 总 SM 的 50%                      │
│    → 保证 HP 至少有 50% 的计算资源                       │
│                                                          │
│  防线 2: Profile 互补                                    │
│    HP=compute 时，只允许 BE=memory                       │
│    HP=memory 时，只允许 BE=compute                       │
│    → 避免争抢同类型资源                                  │
│                                                          │
│  防线 3: 累计时间 (2.5%)                                 │
│    BE 累计执行时间 < HP 延迟的 2.5%                      │
│    → 限制 BE 对 HP 的整体影响                            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 9. 总结

| 组件 | 作用 |
|------|------|
| `run_client()` | 每个客户端的调度循环 |
| `try_pop()` | 非阻塞取操作 |
| `schedule_be()` | 判断 BE 是否允许并发 |
| `is_complementary()` | 检查 HP/BE 资源互补性 |
| `hp_stream_` | 高优先级 CUDA stream |
| `be_streams_[]` | 低优先级 BE streams |
| `hp_task_running_` | 标记 HP 是否在执行 |
| `cumulative_be_duration_ms_` | 累计 BE 执行时间 |

**核心思想**：
- HP 任务：立即执行，优先保证延迟
- BE 任务：在不干扰 HP 的前提下，利用空闲资源
- 通过 profile 互补性和资源阈值，平衡吞吐量和延迟
