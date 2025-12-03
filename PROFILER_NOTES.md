# Profiler 追踪问题说明

## 问题

在使用 PyTorch Profiler 或 nsys 时，所有 `cudaLaunchKernel` 调用都显示在主线程上，
看不到调度器线程。

## 原因

这是 CUPTI (CUDA Profiling Tools Interface) 的工作机制导致的：

1. **CUPTI Callback 机制**：
   - CUPTI 通过在 CUDA runtime/driver 库中注册回调来追踪 API 调用
   - 当使用 LD_PRELOAD 时，CUPTI 的 hook 可能在我们的 wrapper 之外

2. **追踪时机**：
   ```
   应用线程                              调度器线程
       |                                     |
       v                                     |
   调用 cudaLaunchKernel                     |
       |                                     |
       v                                     |
   [CUPTI 记录: 调用开始, tid=主线程]        |
       |                                     |
       v                                     |
   进入我们的 wrapper                        |
       |                                     |
       v                                     |
   操作入队，开始等待 ─────────────────────> 取出操作
       |                                     |
       |                                     v
       |                              调用真正的 cudaLaunchKernel
       |                                     |
       |                                     v
       |                              [GPU 执行 kernel]
       |                                     |
       |                              <───── 操作完成
       v                                     |
   [CUPTI 记录: 调用结束, tid=主线程]        |
       |                                     |
       v                                     |
   wrapper 返回                              |
   ```

3. **PyTorch Profiler 视角**：
   - 它看到的是 "主线程调用了 cudaLaunchKernel 并等待返回"
   - 调度器线程对真正 libcudart 的调用可能被 CUPTI 忽略或无法正确关联

## 验证调度器确实在工作

### 方法 1: 日志验证
```
Client thread=140423659474944 entering WRAPPER, real func at 0x...
Scheduler thread=140417992912896 calling REAL cudaLaunchKernel at 0x...
```
不同的线程 ID 证明是不同的线程在执行。

### 方法 2: 性能验证
```
Without scheduler: 33.91 ms
With scheduler:    33.92 ms
Overhead:          0.0%
```
如果调度器没有真正执行 kernel，性能会有巨大差异。

### 方法 3: 功能验证
GPT 模型推理 3 次迭代全部成功，输出正确。这证明 kernel 被正确执行。

## 结论

- **调度器确实在调度器线程中执行 kernel**
- **GPU kernel 被正确执行**
- **Profiler 显示的线程信息是追踪机制的局限性，不影响实际功能**

## 如何更好地追踪

如果需要在 Profiler 中看到调度器线程：

1. 使用 CUDA Driver API (`cuLaunchKernel`) 而不是 Runtime API
2. 使用 NVTX markers 手动标记调度器线程的活动
3. 使用更底层的追踪工具（如 GPU hardware profiling）
