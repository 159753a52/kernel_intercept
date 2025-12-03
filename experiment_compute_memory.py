#!/usr/bin/env python3
"""
实验：Compute-bound vs Memory-bound 算子并发

设计：
1. Compute workload: 大矩阵乘法 (GEMM) - compute-bound
2. Memory workload: elementwise 操作 - memory-bound

对比：
- Sequential: 先执行所有 compute，再执行所有 memory
- Concurrent: compute 和 memory 交错并发执行
"""

import torch
import time

# 工作负载大小
MATRIX_SIZE = 2048  # 矩阵大小
NUM_OPS = 20        # 每种操作的数量
NUM_WARMUP = 3
NUM_RUNS = 10

def create_compute_workload(size):
    """创建 compute-bound 工作负载（矩阵乘法）"""
    A = torch.randn(size, size, device='cuda')
    B = torch.randn(size, size, device='cuda')
    return A, B

def create_memory_workload(size):
    """创建 memory-bound 工作负载（elementwise 操作）"""
    X = torch.randn(size, size, device='cuda')
    return X

def run_compute_op(A, B):
    """执行 compute-bound 操作"""
    return torch.mm(A, B)

def run_memory_op(X):
    """执行 memory-bound 操作（多个 elementwise 合并）"""
    Y = X + 1
    Y = Y * 2
    Y = torch.relu(Y)
    Y = torch.sigmoid(Y)
    return Y

def benchmark_sequential():
    """顺序执行：先所有 compute，再所有 memory"""
    print("=" * 60)
    print("SEQUENTIAL: All compute first, then all memory")
    print("=" * 60)
    
    # 准备数据
    compute_data = [create_compute_workload(MATRIX_SIZE) for _ in range(NUM_OPS)]
    memory_data = [create_memory_workload(MATRIX_SIZE) for _ in range(NUM_OPS)]
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(NUM_WARMUP):
        for A, B in compute_data:
            _ = run_compute_op(A, B)
        for X in memory_data:
            _ = run_memory_op(X)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 先执行所有 compute 操作
        compute_results = []
        for A, B in compute_data:
            compute_results.append(run_compute_op(A, B))
        
        # 再执行所有 memory 操作
        memory_results = []
        for X in memory_data:
            memory_results.append(run_memory_op(X))
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.2f} ms")
    return avg_time

def benchmark_interleaved():
    """交错执行：compute 和 memory 交替"""
    print("\n" + "=" * 60)
    print("INTERLEAVED: Alternating compute and memory")
    print("=" * 60)
    
    compute_data = [create_compute_workload(MATRIX_SIZE) for _ in range(NUM_OPS)]
    memory_data = [create_memory_workload(MATRIX_SIZE) for _ in range(NUM_OPS)]
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(NUM_WARMUP):
        for i in range(NUM_OPS):
            _ = run_compute_op(*compute_data[i])
            _ = run_memory_op(memory_data[i])
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for i in range(NUM_OPS):
            _ = run_compute_op(*compute_data[i])
            _ = run_memory_op(memory_data[i])
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.2f} ms")
    return avg_time

def benchmark_concurrent_streams():
    """并发执行：compute 和 memory 在不同 stream 上"""
    print("\n" + "=" * 60)
    print("CONCURRENT: Compute and memory on different streams")
    print("=" * 60)
    
    compute_data = [create_compute_workload(MATRIX_SIZE) for _ in range(NUM_OPS)]
    memory_data = [create_memory_workload(MATRIX_SIZE) for _ in range(NUM_OPS)]
    torch.cuda.synchronize()
    
    compute_stream = torch.cuda.Stream()
    memory_stream = torch.cuda.Stream()
    
    # Warmup
    for _ in range(NUM_WARMUP):
        with torch.cuda.stream(compute_stream):
            for A, B in compute_data:
                _ = run_compute_op(A, B)
        with torch.cuda.stream(memory_stream):
            for X in memory_data:
                _ = run_memory_op(X)
        torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        # 同时在两个 stream 上执行
        with torch.cuda.stream(compute_stream):
            for A, B in compute_data:
                _ = run_compute_op(A, B)
        
        with torch.cuda.stream(memory_stream):
            for X in memory_data:
                _ = run_memory_op(X)
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.2f} ms")
    return avg_time

def benchmark_fine_grained_concurrent():
    """细粒度并发：每对 compute+memory 操作并发"""
    print("\n" + "=" * 60)
    print("FINE-GRAINED CONCURRENT: Each compute+memory pair concurrent")
    print("=" * 60)
    
    compute_data = [create_compute_workload(MATRIX_SIZE) for _ in range(NUM_OPS)]
    memory_data = [create_memory_workload(MATRIX_SIZE) for _ in range(NUM_OPS)]
    torch.cuda.synchronize()
    
    compute_stream = torch.cuda.Stream()
    memory_stream = torch.cuda.Stream()
    
    # Warmup
    for _ in range(NUM_WARMUP):
        for i in range(NUM_OPS):
            with torch.cuda.stream(compute_stream):
                _ = run_compute_op(*compute_data[i])
            with torch.cuda.stream(memory_stream):
                _ = run_memory_op(memory_data[i])
        torch.cuda.synchronize()
    
    # Benchmark  
    times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        for i in range(NUM_OPS):
            # 每次一个 compute 和一个 memory 操作并发
            with torch.cuda.stream(compute_stream):
                _ = run_compute_op(*compute_data[i])
            with torch.cuda.stream(memory_stream):
                _ = run_memory_op(memory_data[i])
        
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    avg_time = sum(times) / len(times)
    print(f"Average time: {avg_time:.2f} ms")
    return avg_time

def main():
    print("=" * 60)
    print("EXPERIMENT: Compute + Memory Bound Concurrency")
    print("=" * 60)
    print(f"Matrix size: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Ops per type: {NUM_OPS}")
    print(f"Runs: {NUM_RUNS}")
    print()
    
    time_seq = benchmark_sequential()
    time_inter = benchmark_interleaved()
    time_concurrent = benchmark_concurrent_streams()
    time_fine = benchmark_fine_grained_concurrent()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequential (baseline):     {time_seq:.2f} ms")
    print(f"Interleaved:               {time_inter:.2f} ms  (speedup: {time_seq/time_inter:.2f}x)")
    print(f"Concurrent (2 streams):    {time_concurrent:.2f} ms  (speedup: {time_seq/time_concurrent:.2f}x)")
    print(f"Fine-grained concurrent:   {time_fine:.2f} ms  (speedup: {time_seq/time_fine:.2f}x)")

if __name__ == "__main__":
    main()
