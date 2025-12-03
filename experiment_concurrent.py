#!/usr/bin/env python3
"""
实验：对比顺序执行 vs 并发执行（compute + memory bound 算子）

实验设计：
1. Baseline: 顺序执行两次推理
2. Optimized: 两个推理任务分配到不同 stream，让互补算子并发
"""

import torch
import time
import sys
import os

sys.path.insert(0, ".")
from GPT import create_char_gpt, sequence_len

NUM_WARMUP = 5
NUM_RUNS = 10

def benchmark_sequential():
    """顺序执行两次推理"""
    print("=" * 60)
    print("SEQUENTIAL EXECUTION (baseline)")
    print("=" * 60)
    
    model = create_char_gpt(256, "cuda")
    model.eval()
    
    input1 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    input2 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    
    # Warmup
    print("Warmup...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            _ = model(input1)
            _ = model(input2)
    torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {NUM_RUNS} iterations...")
    times = []
    for i in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            out1 = model(input1)  # 第一次推理
            out2 = model(input2)  # 第二次推理
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    
    del model
    torch.cuda.empty_cache()
    
    return avg_time

def benchmark_concurrent_streams():
    """使用两个 stream 并发执行"""
    print("\n" + "=" * 60)
    print("CONCURRENT EXECUTION (two streams)")
    print("=" * 60)
    
    model = create_char_gpt(256, "cuda")
    model.eval()
    
    input1 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    input2 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    
    # 创建两个 stream
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # Warmup
    print("Warmup...")
    for _ in range(NUM_WARMUP):
        with torch.no_grad():
            with torch.cuda.stream(stream1):
                _ = model(input1)
            with torch.cuda.stream(stream2):
                _ = model(input2)
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {NUM_RUNS} iterations...")
    times = []
    for i in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.no_grad():
            # 两个推理在不同 stream 上执行
            with torch.cuda.stream(stream1):
                out1 = model(input1)
            with torch.cuda.stream(stream2):
                out2 = model(input2)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    
    del model
    torch.cuda.empty_cache()
    
    return avg_time

def benchmark_with_scheduler():
    """使用调度器，两个客户端线程"""
    import ctypes
    import threading
    
    print("\n" + "=" * 60)
    print("SCHEDULED EXECUTION (scheduler with 2 clients)")
    print("=" * 60)
    
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(2)  # 2 个客户端
    
    # Passthrough 模式创建模型
    model = create_char_gpt(256, "cuda")
    model.eval()
    
    input1 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    input2 = torch.randint(0, 256, (2, sequence_len), device="cuda")
    torch.cuda.synchronize()
    
    results = [None, None]
    
    def client_task(client_idx, input_tensor):
        lib.orion_set_client_idx(client_idx)
        with torch.no_grad():
            output = model(input_tensor)
        torch.cuda.synchronize()
        results[client_idx] = output
    
    # Warmup
    print("Warmup...")
    for _ in range(NUM_WARMUP):
        t1 = threading.Thread(target=client_task, args=(0, input1))
        t2 = threading.Thread(target=client_task, args=(1, input2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    
    # Benchmark
    print(f"Running {NUM_RUNS} iterations...")
    times = []
    for i in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        t1 = threading.Thread(target=client_task, args=(0, input1))
        t2 = threading.Thread(target=client_task, args=(1, input2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"\nResults:")
    print(f"  Average: {avg_time:.2f} ms")
    print(f"  Min: {min_time:.2f} ms")
    print(f"  Max: {max_time:.2f} ms")
    
    lib.orion_stop_scheduler()
    del model
    torch.cuda.empty_cache()
    
    return avg_time

def main():
    print("=" * 60)
    print("EXPERIMENT: Sequential vs Concurrent Execution")
    print("=" * 60)
    print(f"Model: GPT")
    print(f"Warmup iterations: {NUM_WARMUP}")
    print(f"Benchmark iterations: {NUM_RUNS}")
    print()
    
    # 1. 顺序执行
    time_seq = benchmark_sequential()
    
    # 2. 使用两个 stream 并发
    time_streams = benchmark_concurrent_streams()
    
    # 3. 使用调度器
    # 注意：需要 LD_PRELOAD
    use_scheduler = "libgpu_scheduler.so" in os.environ.get("LD_PRELOAD", "")
    if use_scheduler:
        time_sched = benchmark_with_scheduler()
    else:
        time_sched = None
        print("\n[SKIP] Scheduler benchmark (run with LD_PRELOAD)")
    
    # 总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequential (baseline):    {time_seq:.2f} ms")
    print(f"Concurrent (2 streams):   {time_streams:.2f} ms  (speedup: {time_seq/time_streams:.2f}x)")
    if time_sched:
        print(f"Scheduled (2 clients):    {time_sched:.2f} ms  (speedup: {time_seq/time_sched:.2f}x)")

if __name__ == "__main__":
    main()
