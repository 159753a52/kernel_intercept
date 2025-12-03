#!/usr/bin/env python3
"""
实验：量化调度器各部分开销

测量：
1. 原生 CUDA 调用开销
2. LD_PRELOAD wrapper 开销
3. 队列操作开销
4. 条件变量等待开销
5. 总体调度器开销
"""

import torch
import time
import ctypes
import os
import sys

NUM_OPS = 10000  # 操作次数
NUM_RUNS = 5     # 重复次数

def benchmark_native_cuda():
    """原生 CUDA 操作（无拦截）"""
    print("=" * 60)
    print("1. NATIVE CUDA (no interception)")
    print("=" * 60)
    
    A = torch.randn(256, 256, device='cuda')
    B = torch.randn(256, 256, device='cuda')
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(100):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(NUM_OPS):
            C = torch.mm(A, B)
        
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    avg = sum(times) / len(times)
    per_op = avg / NUM_OPS * 1000  # us per op
    print(f"Total time: {avg:.2f} ms")
    print(f"Per operation: {per_op:.3f} us")
    return avg, per_op

def benchmark_with_preload_passthrough():
    """LD_PRELOAD 但 passthrough 模式"""
    print("\n" + "=" * 60)
    print("2. LD_PRELOAD PASSTHROUGH (wrapper overhead only)")
    print("=" * 60)
    
    if "libgpu_scheduler.so" not in os.environ.get("LD_PRELOAD", ""):
        print("[SKIP] Run with LD_PRELOAD")
        return None, None
    
    # 不设置 client_idx，保持 passthrough 模式
    A = torch.randn(256, 256, device='cuda')
    B = torch.randn(256, 256, device='cuda')
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(100):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(NUM_OPS):
            C = torch.mm(A, B)
        
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    avg = sum(times) / len(times)
    per_op = avg / NUM_OPS * 1000
    print(f"Total time: {avg:.2f} ms")
    print(f"Per operation: {per_op:.3f} us")
    return avg, per_op

def benchmark_with_scheduler():
    """完整调度器"""
    print("\n" + "=" * 60)
    print("3. FULL SCHEDULER (queue + wait)")
    print("=" * 60)
    
    if "libgpu_scheduler.so" not in os.environ.get("LD_PRELOAD", ""):
        print("[SKIP] Run with LD_PRELOAD")
        return None, None
    
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(1)
    lib.orion_set_client_idx(0)
    
    A = torch.randn(256, 256, device='cuda')
    B = torch.randn(256, 256, device='cuda')
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(100):
        C = torch.mm(A, B)
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(NUM_OPS):
            C = torch.mm(A, B)
        
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    
    avg = sum(times) / len(times)
    per_op = avg / NUM_OPS * 1000
    print(f"Total time: {avg:.2f} ms")
    print(f"Per operation: {per_op:.3f} us")
    
    lib.orion_stop_scheduler()
    return avg, per_op

def benchmark_queue_only():
    """仅测量队列操作（Python 模拟）"""
    print("\n" + "=" * 60)
    print("4. QUEUE OPERATIONS ONLY (Python simulation)")
    print("=" * 60)
    
    import queue
    import threading
    
    q = queue.Queue()
    results = queue.Queue()
    running = True
    
    def consumer():
        while running:
            try:
                item = q.get(timeout=0.001)
                results.put(item)
                q.task_done()
            except queue.Empty:
                pass
    
    thread = threading.Thread(target=consumer)
    thread.start()
    
    # Warmup
    for i in range(1000):
        q.put(i)
        results.get()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        
        for i in range(NUM_OPS):
            q.put(i)
            results.get()
        
        times.append((time.perf_counter() - start) * 1000)
    
    running = False
    thread.join()
    
    avg = sum(times) / len(times)
    per_op = avg / NUM_OPS * 1000
    print(f"Total time: {avg:.2f} ms")
    print(f"Per operation: {per_op:.3f} us")
    return avg, per_op

def benchmark_condition_variable():
    """仅测量条件变量开销"""
    print("\n" + "=" * 60)
    print("5. CONDITION VARIABLE ONLY")
    print("=" * 60)
    
    import threading
    
    cv = threading.Condition()
    ready = False
    done = False
    running = True
    
    def worker():
        nonlocal done
        while running:
            with cv:
                cv.wait_for(lambda: ready or not running)
                if not running:
                    break
                done = True
                cv.notify()
    
    thread = threading.Thread(target=worker)
    thread.start()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        nonlocal_vars = {'ready': False, 'done': False}
        start = time.perf_counter()
        
        for i in range(NUM_OPS):
            with cv:
                nonlocal_vars['ready'] = True
                cv.notify()
            with cv:
                cv.wait_for(lambda: nonlocal_vars['done'])
                nonlocal_vars['ready'] = False
                nonlocal_vars['done'] = False
        
        times.append((time.perf_counter() - start) * 1000)
    
    running = False
    with cv:
        cv.notify()
    thread.join()
    
    avg = sum(times) / len(times)
    per_op = avg / NUM_OPS * 1000
    print(f"Total time: {avg:.2f} ms")
    print(f"Per operation: {per_op:.3f} us")
    return avg, per_op

def main():
    print("=" * 60)
    print("SCHEDULER OVERHEAD ANALYSIS")
    print("=" * 60)
    print(f"Operations: {NUM_OPS}")
    print(f"Runs: {NUM_RUNS}")
    print()
    
    results = {}
    
    # 原生 CUDA
    native_total, native_per_op = benchmark_native_cuda()
    results['native'] = native_per_op
    
    # Passthrough
    pt_total, pt_per_op = benchmark_with_preload_passthrough()
    if pt_per_op:
        results['passthrough'] = pt_per_op
    
    # 完整调度器
    sched_total, sched_per_op = benchmark_with_scheduler()
    if sched_per_op:
        results['scheduler'] = sched_per_op
    
    # Python 队列模拟
    q_total, q_per_op = benchmark_queue_only()
    results['queue_python'] = q_per_op
    
    # 总结
    print("\n" + "=" * 60)
    print("OVERHEAD BREAKDOWN (per operation)")
    print("=" * 60)
    
    print(f"\n1. Native CUDA call:        {results['native']:.3f} us")
    
    if 'passthrough' in results:
        wrapper_overhead = results['passthrough'] - results['native']
        print(f"2. LD_PRELOAD wrapper:      {results['passthrough']:.3f} us (+{wrapper_overhead:.3f} us)")
    
    if 'scheduler' in results:
        total_overhead = results['scheduler'] - results['native']
        print(f"3. Full scheduler:          {results['scheduler']:.3f} us (+{total_overhead:.3f} us)")
        
        if 'passthrough' in results:
            queue_overhead = results['scheduler'] - results['passthrough']
            print(f"\n   Breakdown:")
            print(f"   - Wrapper overhead:      +{wrapper_overhead:.3f} us")
            print(f"   - Queue/sync overhead:   +{queue_overhead:.3f} us")
    
    print(f"\n4. Python queue (reference): {results['queue_python']:.3f} us")

if __name__ == "__main__":
    main()
