#!/usr/bin/env python3
"""
实验：GPT (compute-bound) + Memory workload (bandwidth-bound) 并发

Client 0: GPT 推理 (compute-bound, 大量 GEMM)
Client 1: 带宽受限任务 (大规模 elementwise + memcpy)
"""

import torch
import time
import ctypes
import threading
import os
import sys

sys.path.insert(0, ".")
from GPT import create_char_gpt, sequence_len

# 带宽受限任务的参数
MEMORY_SIZE = 4096  # 大矩阵
MEMORY_OPS = 50     # 重复次数

def create_memory_workload():
    """创建带宽受限的工作负载"""
    tensors = [torch.randn(MEMORY_SIZE, MEMORY_SIZE, device='cuda') for _ in range(5)]
    return tensors

def run_memory_workload(tensors):
    """执行带宽受限的操作"""
    for _ in range(MEMORY_OPS):
        # Elementwise 操作 - memory bound
        x = tensors[0] + tensors[1]
        x = x * tensors[2]
        x = torch.relu(x)
        x = x + tensors[3]
        x = torch.sigmoid(x)
        # 数据拷贝
        tensors[4].copy_(x)
    return tensors[4]

def benchmark_sequential():
    """顺序执行：先 GPT，再 Memory workload"""
    print("=" * 60)
    print("SEQUENTIAL: GPT first, then Memory workload")
    print("=" * 60)
    
    model = create_char_gpt(256, "cuda")
    model.eval()
    gpt_input = torch.randint(0, 256, (2, sequence_len), device="cuda")
    mem_tensors = create_memory_workload()
    torch.cuda.synchronize()
    
    # Warmup
    with torch.no_grad():
        _ = model(gpt_input)
    _ = run_memory_workload(mem_tensors)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.no_grad():
        gpt_out = model(gpt_input)
    mem_out = run_memory_workload(mem_tensors)
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"Total time: {total_time:.2f} ms")
    
    del model
    torch.cuda.empty_cache()
    return total_time

def benchmark_native_streams():
    """原生 PyTorch：两个 stream 并发"""
    print("\n" + "=" * 60)
    print("NATIVE STREAMS: GPT and Memory on different streams")
    print("=" * 60)
    
    model = create_char_gpt(256, "cuda")
    model.eval()
    gpt_input = torch.randint(0, 256, (2, sequence_len), device="cuda")
    mem_tensors = create_memory_workload()
    torch.cuda.synchronize()
    
    stream_gpt = torch.cuda.Stream()
    stream_mem = torch.cuda.Stream()
    
    # Warmup
    with torch.cuda.stream(stream_gpt):
        with torch.no_grad():
            _ = model(gpt_input)
    with torch.cuda.stream(stream_mem):
        _ = run_memory_workload(mem_tensors)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    with torch.cuda.stream(stream_gpt):
        with torch.no_grad():
            gpt_out = model(gpt_input)
    
    with torch.cuda.stream(stream_mem):
        mem_out = run_memory_workload(mem_tensors)
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"Total time: {total_time:.2f} ms")
    
    del model
    torch.cuda.empty_cache()
    return total_time

def benchmark_threads_streams():
    """两个线程 + 两个 stream"""
    print("\n" + "=" * 60)
    print("TWO THREADS: Each with its own stream")
    print("=" * 60)
    
    model = create_char_gpt(256, "cuda")
    model.eval()
    gpt_input = torch.randint(0, 256, (2, sequence_len), device="cuda")
    mem_tensors = create_memory_workload()
    torch.cuda.synchronize()
    
    stream_gpt = torch.cuda.Stream()
    stream_mem = torch.cuda.Stream()
    
    def gpt_worker():
        with torch.cuda.stream(stream_gpt):
            with torch.no_grad():
                _ = model(gpt_input)
    
    def mem_worker():
        with torch.cuda.stream(stream_mem):
            _ = run_memory_workload(mem_tensors)
    
    # Warmup
    t1 = threading.Thread(target=gpt_worker)
    t2 = threading.Thread(target=mem_worker)
    t1.start(); t2.start()
    t1.join(); t2.join()
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    t1 = threading.Thread(target=gpt_worker)
    t2 = threading.Thread(target=mem_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"Total time: {total_time:.2f} ms")
    
    del model
    torch.cuda.empty_cache()
    return total_time

def benchmark_scheduler():
    """调度器：两个客户端"""
    print("\n" + "=" * 60)
    print("SCHEDULER: GPT (client 0) + Memory (client 1)")
    print("=" * 60)
    
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(2)
    
    model = create_char_gpt(256, "cuda")
    model.eval()
    gpt_input = torch.randint(0, 256, (2, sequence_len), device="cuda")
    mem_tensors = create_memory_workload()
    torch.cuda.synchronize()
    
    def gpt_client():
        lib.orion_set_client_idx(0)
        with torch.no_grad():
            _ = model(gpt_input)
        torch.cuda.synchronize()
    
    def mem_client():
        lib.orion_set_client_idx(1)
        _ = run_memory_workload(mem_tensors)
        torch.cuda.synchronize()
    
    # Warmup
    t1 = threading.Thread(target=gpt_client)
    t2 = threading.Thread(target=mem_client)
    t1.start(); t2.start()
    t1.join(); t2.join()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    t1 = threading.Thread(target=gpt_client)
    t2 = threading.Thread(target=mem_client)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"Total time: {total_time:.2f} ms")
    
    lib.orion_stop_scheduler()
    del model
    torch.cuda.empty_cache()
    return total_time

def main():
    print("=" * 60)
    print("EXPERIMENT: GPT (Compute) + Memory Workload Concurrency")
    print("=" * 60)
    print(f"GPT model: 256 vocab, seq_len={sequence_len}")
    print(f"Memory workload: {MEMORY_SIZE}x{MEMORY_SIZE} tensors, {MEMORY_OPS} iterations")
    print()
    
    time_seq = benchmark_sequential()
    time_native = benchmark_native_streams()
    time_threads = benchmark_threads_streams()
    
    if "libgpu_scheduler.so" in os.environ.get("LD_PRELOAD", ""):
        time_sched = benchmark_scheduler()
    else:
        time_sched = None
        print("\n[SKIP] Scheduler - run with LD_PRELOAD")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequential (baseline):    {time_seq:.2f} ms")
    print(f"Native streams:           {time_native:.2f} ms  (speedup: {time_seq/time_native:.2f}x)")
    print(f"Two threads + streams:    {time_threads:.2f} ms  (speedup: {time_seq/time_threads:.2f}x)")
    if time_sched:
        print(f"Scheduler (2 clients):    {time_sched:.2f} ms  (speedup: {time_seq/time_sched:.2f}x)")

if __name__ == "__main__":
    main()
