#!/usr/bin/env python3
"""
实验：两个客户端线程，各自执行一次 GPT 推理

对比：
1. Sequential: 单线程顺序执行两次推理
2. Two threads + Two streams: 两个线程各自用自己的 stream
3. Scheduler: 两个客户端线程，调度器分配 stream
"""

import torch
import time
import threading
import os
import sys

sys.path.insert(0, ".")
from GPT import create_char_gpt, sequence_len

NUM_WARMUP = 3
NUM_RUNS = 10

def benchmark_sequential():
    """单线程顺序执行两次推理"""
    print("=" * 60)
    print("SEQUENTIAL: Single thread, two inferences")
    print("=" * 60)
    
    model = create_char_gpt(256, "cuda")
    model.eval()
    
    input1 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    input2 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = model(input1)
            _ = model(input2)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            out1 = model(input1)
            out2 = model(input2)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg = sum(times) / len(times)
    print(f"Average time: {avg:.2f} ms")
    
    del model
    torch.cuda.empty_cache()
    return avg

def benchmark_two_threads_two_streams():
    """两个线程，各自用自己的 stream"""
    print("\n" + "=" * 60)
    print("TWO THREADS + TWO STREAMS: Each thread has its own stream")
    print("=" * 60)
    
    model = create_char_gpt(256, "cuda")
    model.eval()
    
    input1 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    input2 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    torch.cuda.synchronize()
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    results = [None, None]
    
    def worker1():
        with torch.cuda.stream(stream1):
            with torch.no_grad():
                results[0] = model(input1)
    
    def worker2():
        with torch.cuda.stream(stream2):
            with torch.no_grad():
                results[1] = model(input2)
    
    # Warmup
    for _ in range(NUM_WARMUP):
        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        t1 = threading.Thread(target=worker1)
        t2 = threading.Thread(target=worker2)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg = sum(times) / len(times)
    print(f"Average time: {avg:.2f} ms")
    
    del model
    torch.cuda.empty_cache()
    return avg

def benchmark_with_scheduler():
    """使用调度器：两个客户端线程"""
    import ctypes
    
    print("\n" + "=" * 60)
    print("SCHEDULER: Two client threads, scheduler assigns streams")
    print("=" * 60)
    
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(2)
    
    # Passthrough 模式创建模型
    model = create_char_gpt(256, "cuda")
    model.eval()
    
    input1 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    input2 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    torch.cuda.synchronize()
    
    results = [None, None]
    
    def client0():
        lib.orion_set_client_idx(0)
        with torch.no_grad():
            results[0] = model(input1)
        torch.cuda.synchronize()
    
    def client1():
        lib.orion_set_client_idx(1)
        with torch.no_grad():
            results[1] = model(input2)
        torch.cuda.synchronize()
    
    # Warmup
    for _ in range(NUM_WARMUP):
        t1 = threading.Thread(target=client0)
        t2 = threading.Thread(target=client1)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        t1 = threading.Thread(target=client0)
        t2 = threading.Thread(target=client1)
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
    del model
    torch.cuda.empty_cache()
    return avg

def main():
    print("=" * 60)
    print("EXPERIMENT: Two GPT Inferences with Two Client Threads")
    print("=" * 60)
    print()
    
    time_seq = benchmark_sequential()
    time_threads = benchmark_two_threads_two_streams()
    
    if "libgpu_scheduler.so" in os.environ.get("LD_PRELOAD", ""):
        time_sched = benchmark_with_scheduler()
    else:
        time_sched = None
        print("\n[SKIP] Scheduler - run with LD_PRELOAD")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequential (baseline):      {time_seq:.2f} ms")
    print(f"Two threads + streams:      {time_threads:.2f} ms  (speedup: {time_seq/time_threads:.2f}x)")
    if time_sched:
        print(f"Scheduler (2 clients):      {time_sched:.2f} ms  (speedup: {time_seq/time_sched:.2f}x)")

if __name__ == "__main__":
    main()
