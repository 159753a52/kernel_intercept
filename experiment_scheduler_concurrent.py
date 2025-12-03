#!/usr/bin/env python3
"""
实验：使用调度器实现 Compute/Memory 并发

方案：
- Client 0: 执行 compute-bound 操作 (矩阵乘法)
- Client 1: 执行 memory-bound 操作 (elementwise)
- 两个客户端使用不同的 stream，实现并发
"""

import torch
import time
import ctypes
import threading
import os
import sys

MATRIX_SIZE = 2048
NUM_OPS = 20
NUM_WARMUP = 3
NUM_RUNS = 10

def create_compute_data():
    """Compute-bound 数据"""
    return [(torch.randn(MATRIX_SIZE, MATRIX_SIZE, device='cuda'),
             torch.randn(MATRIX_SIZE, MATRIX_SIZE, device='cuda')) 
            for _ in range(NUM_OPS)]

def create_memory_data():
    """Memory-bound 数据"""
    return [torch.randn(MATRIX_SIZE, MATRIX_SIZE, device='cuda') 
            for _ in range(NUM_OPS)]

def run_compute_ops(data):
    """执行所有 compute 操作"""
    results = []
    for A, B in data:
        results.append(torch.mm(A, B))
    return results

def run_memory_ops(data):
    """执行所有 memory 操作"""
    results = []
    for X in data:
        Y = X + 1
        Y = Y * 2
        Y = torch.relu(Y)
        Y = torch.sigmoid(Y)
        results.append(Y)
    return results

def benchmark_sequential():
    """顺序执行 (baseline)"""
    print("=" * 60)
    print("BASELINE: Sequential execution (no scheduler)")
    print("=" * 60)
    
    compute_data = create_compute_data()
    memory_data = create_memory_data()
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(NUM_WARMUP):
        run_compute_ops(compute_data)
        run_memory_ops(memory_data)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        run_compute_ops(compute_data)
        run_memory_ops(memory_data)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg = sum(times) / len(times)
    print(f"Average time: {avg:.2f} ms")
    return avg

def benchmark_with_scheduler():
    """使用调度器并发执行"""
    print("\n" + "=" * 60)
    print("SCHEDULED: Compute and Memory on separate streams")
    print("=" * 60)
    
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(2)  # 2 clients
    
    compute_data = create_compute_data()
    memory_data = create_memory_data()
    torch.cuda.synchronize()
    
    results = [None, None]
    done_events = [threading.Event(), threading.Event()]
    
    def compute_client():
        """Client 0: Compute operations"""
        lib.orion_set_client_idx(0)
        results[0] = run_compute_ops(compute_data)
        torch.cuda.synchronize()
        done_events[0].set()
    
    def memory_client():
        """Client 1: Memory operations"""
        lib.orion_set_client_idx(1)
        results[1] = run_memory_ops(memory_data)
        torch.cuda.synchronize()
        done_events[1].set()
    
    # Warmup
    for _ in range(NUM_WARMUP):
        done_events[0].clear()
        done_events[1].clear()
        t1 = threading.Thread(target=compute_client)
        t2 = threading.Thread(target=memory_client)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        done_events[0].clear()
        done_events[1].clear()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        t1 = threading.Thread(target=compute_client)
        t2 = threading.Thread(target=memory_client)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg = sum(times) / len(times)
    print(f"Average time: {avg:.2f} ms")
    
    lib.orion_stop_scheduler()
    return avg

def benchmark_pytorch_streams():
    """使用 PyTorch streams 并发（对照）"""
    print("\n" + "=" * 60)
    print("PYTORCH STREAMS: Native concurrent execution")
    print("=" * 60)
    
    compute_data = create_compute_data()
    memory_data = create_memory_data()
    torch.cuda.synchronize()
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # Warmup
    for _ in range(NUM_WARMUP):
        with torch.cuda.stream(stream1):
            run_compute_ops(compute_data)
        with torch.cuda.stream(stream2):
            run_memory_ops(memory_data)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.cuda.stream(stream1):
            run_compute_ops(compute_data)
        with torch.cuda.stream(stream2):
            run_memory_ops(memory_data)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg = sum(times) / len(times)
    print(f"Average time: {avg:.2f} ms")
    return avg

def main():
    print("=" * 60)
    print("EXPERIMENT: Scheduler-based Compute/Memory Concurrency")
    print("=" * 60)
    print(f"Matrix size: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Operations per type: {NUM_OPS}")
    print()
    
    time_seq = benchmark_sequential()
    time_pytorch = benchmark_pytorch_streams()
    
    # 使用调度器（需要 LD_PRELOAD）
    if "libgpu_scheduler.so" in os.environ.get("LD_PRELOAD", ""):
        time_sched = benchmark_with_scheduler()
    else:
        time_sched = None
        print("\n[SKIP] Scheduler benchmark - run with LD_PRELOAD")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequential (baseline):    {time_seq:.2f} ms")
    print(f"PyTorch streams:          {time_pytorch:.2f} ms  (speedup: {time_seq/time_pytorch:.2f}x)")
    if time_sched:
        print(f"Scheduler (2 clients):    {time_sched:.2f} ms  (speedup: {time_seq/time_sched:.2f}x)")

if __name__ == "__main__":
    main()
