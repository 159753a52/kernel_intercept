#!/usr/bin/env python3
"""
实验：大规模任务下的调度器开销分析

增大任务规模，让任务执行时间远大于同步开销
"""

import torch
import time
import ctypes
import threading
import os
import sys

sys.path.insert(0, ".")
from GPT import create_char_gpt, sequence_len

def create_large_compute_task():
    """大规模计算任务：多次大矩阵乘法"""
    SIZE = 4096  # 大矩阵
    NUM_OPS = 100  # 多次操作
    
    A = torch.randn(SIZE, SIZE, device='cuda')
    B = torch.randn(SIZE, SIZE, device='cuda')
    
    def run():
        for _ in range(NUM_OPS):
            C = torch.mm(A, B)
        return C
    
    return run, f"GEMM {SIZE}x{SIZE} x {NUM_OPS}"

def create_large_memory_task():
    """大规模带宽任务：多次 elementwise 操作"""
    SIZE = 8192  # 更大的矩阵（带宽受限不需要那么多计算）
    NUM_OPS = 200  # 更多次操作
    
    tensors = [torch.randn(SIZE, SIZE, device='cuda') for _ in range(5)]
    
    def run():
        for _ in range(NUM_OPS):
            x = tensors[0] + tensors[1]
            x = x * tensors[2]
            x = torch.relu(x)
            x = x + tensors[3]
            x = torch.sigmoid(x)
            tensors[4].copy_(x)
        return tensors[4]
    
    return run, f"Elementwise {SIZE}x{SIZE} x {NUM_OPS}"

def create_gpt_task(num_iters=5):
    """GPT 推理任务"""
    model = create_char_gpt(256, "cuda")
    model.eval()
    input_tensor = torch.randint(0, 256, (4, sequence_len), device="cuda")  # batch=4
    
    def run():
        for _ in range(num_iters):
            with torch.no_grad():
                out = model(input_tensor)
        return out
    
    return run, model, f"GPT inference x {num_iters}"

def benchmark_sequential(compute_task, memory_task, compute_desc, memory_desc):
    """顺序执行"""
    print("=" * 60)
    print(f"SEQUENTIAL: {compute_desc} + {memory_desc}")
    print("=" * 60)
    
    # Warmup
    compute_task()
    memory_task()
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    compute_task()
    memory_task()
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"Total time: {total_time:.2f} ms")
    return total_time

def benchmark_native_concurrent(compute_task, memory_task, compute_desc, memory_desc):
    """原生两线程+两stream并发"""
    print("\n" + "=" * 60)
    print(f"NATIVE CONCURRENT: {compute_desc} || {memory_desc}")
    print("=" * 60)
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    def compute_worker():
        with torch.cuda.stream(stream1):
            compute_task()
    
    def memory_worker():
        with torch.cuda.stream(stream2):
            memory_task()
    
    # Warmup
    t1 = threading.Thread(target=compute_worker)
    t2 = threading.Thread(target=memory_worker)
    t1.start(); t2.start()
    t1.join(); t2.join()
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    t1 = threading.Thread(target=compute_worker)
    t2 = threading.Thread(target=memory_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"Total time: {total_time:.2f} ms")
    return total_time

def benchmark_scheduler(compute_task, memory_task, compute_desc, memory_desc):
    """调度器并发"""
    print("\n" + "=" * 60)
    print(f"SCHEDULER: {compute_desc} || {memory_desc}")
    print("=" * 60)
    
    if "libgpu_scheduler.so" not in os.environ.get("LD_PRELOAD", ""):
        print("[SKIP] Run with LD_PRELOAD")
        return None
    
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(2)
    
    def compute_client():
        lib.orion_set_client_idx(0)
        compute_task()
        torch.cuda.synchronize()
    
    def memory_client():
        lib.orion_set_client_idx(1)
        memory_task()
        torch.cuda.synchronize()
    
    # Warmup
    t1 = threading.Thread(target=compute_client)
    t2 = threading.Thread(target=memory_client)
    t1.start(); t2.start()
    t1.join(); t2.join()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    t1 = threading.Thread(target=compute_client)
    t2 = threading.Thread(target=memory_client)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"Total time: {total_time:.2f} ms")
    
    lib.orion_stop_scheduler()
    return total_time

def run_experiment(name, compute_task, memory_task, compute_desc, memory_desc):
    """运行一组实验"""
    print("\n" + "#" * 60)
    print(f"# EXPERIMENT: {name}")
    print("#" * 60 + "\n")
    
    time_seq = benchmark_sequential(compute_task, memory_task, compute_desc, memory_desc)
    time_native = benchmark_native_concurrent(compute_task, memory_task, compute_desc, memory_desc)
    time_sched = benchmark_scheduler(compute_task, memory_task, compute_desc, memory_desc)
    
    print("\n" + "-" * 40)
    print("RESULTS:")
    print("-" * 40)
    print(f"Sequential:        {time_seq:.2f} ms")
    print(f"Native concurrent: {time_native:.2f} ms  (speedup: {time_seq/time_native:.2f}x)")
    if time_sched:
        print(f"Scheduler:         {time_sched:.2f} ms  (speedup: {time_seq/time_sched:.2f}x)")
    
    return time_seq, time_native, time_sched

def main():
    print("=" * 60)
    print("LARGE TASK EXPERIMENT")
    print("=" * 60)
    print("Testing if scheduler overhead can be amortized by large tasks\n")
    
    results = []
    
    # 实验1：大规模 GEMM + Elementwise
    compute_task, compute_desc = create_large_compute_task()
    memory_task, memory_desc = create_large_memory_task()
    torch.cuda.synchronize()
    
    r = run_experiment("Large GEMM + Large Elementwise", 
                       compute_task, memory_task, compute_desc, memory_desc)
    results.append(("GEMM + Elem", r))
    
    # 清理
    torch.cuda.empty_cache()
    
    # 实验2：GPT + Elementwise
    gpt_task, model, gpt_desc = create_gpt_task(num_iters=10)
    memory_task2, memory_desc2 = create_large_memory_task()
    torch.cuda.synchronize()
    
    r = run_experiment("GPT Inference + Large Elementwise",
                       gpt_task, memory_task2, gpt_desc, memory_desc2)
    results.append(("GPT + Elem", r))
    
    del model
    torch.cuda.empty_cache()
    
    # 总结
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Experiment':<20} {'Sequential':>12} {'Native':>12} {'Scheduler':>12}")
    print("-" * 60)
    for name, (seq, native, sched) in results:
        sched_str = f"{sched:.1f} ms" if sched else "N/A"
        print(f"{name:<20} {seq:>10.1f} ms {native:>10.1f} ms {sched_str:>12}")

if __name__ == "__main__":
    main()
