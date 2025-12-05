#!/usr/bin/env python3
"""
实验：分块 GEMM 是否能让其他 kernel 并发执行？

对比：
1. 完整 GEMM + Memory 操作（顺序）
2. 分块 GEMM + Memory 操作（交替执行，创造并发机会）
"""

import torch
import time
import threading

M, N, K = 4096, 4096, 4096
K_TILE = 512
NUM_CHUNKS = K // K_TILE

def chunked_gemm_with_callback(A, B, callback=None, k_tile=K_TILE):
    """分块 GEMM，每个 chunk 后可以执行回调"""
    M, K = A.shape
    _, N = B.shape
    C = torch.zeros(M, N, device=A.device, dtype=A.dtype)
    
    for k in range(0, K, k_tile):
        k_end = min(k + k_tile, K)
        A_chunk = A[:, k:k_end]
        B_chunk = B[k:k_end, :]
        
        if k == 0:
            C = torch.mm(A_chunk, B_chunk)
        else:
            C = C + torch.mm(A_chunk, B_chunk)
        
        # 每个 chunk 后的回调（可以插入其他操作）
        if callback:
            callback(k // k_tile)
    
    return C

def memory_operation(tensors, idx):
    """Memory-bound 操作"""
    x = tensors[0] + tensors[1]
    x = x * tensors[2]
    x = torch.relu(x)
    return x

def benchmark_sequential():
    """顺序执行：先完整 GEMM，再 Memory 操作"""
    print("=" * 60)
    print("1. SEQUENTIAL: Full GEMM, then Memory ops")
    print("=" * 60)
    
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    tensors = [torch.randn(2048, 2048, device='cuda') for _ in range(3)]
    torch.cuda.synchronize()
    
    # Warmup
    _ = torch.mm(A, B)
    for _ in range(NUM_CHUNKS):
        _ = memory_operation(tensors, 0)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # 先执行完整 GEMM
    C = torch.mm(A, B)
    # 再执行 Memory 操作
    for i in range(NUM_CHUNKS):
        _ = memory_operation(tensors, i)
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"  Total time: {total_time:.2f} ms")
    return total_time

def benchmark_interleaved():
    """交替执行：每个 GEMM chunk 后插入 Memory 操作"""
    print("\n" + "=" * 60)
    print("2. INTERLEAVED: GEMM chunk, then Memory op, repeat")
    print("=" * 60)
    
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    tensors = [torch.randn(2048, 2048, device='cuda') for _ in range(3)]
    torch.cuda.synchronize()
    
    # Warmup
    def dummy_callback(i):
        _ = memory_operation(tensors, i)
    _ = chunked_gemm_with_callback(A, B, dummy_callback, K_TILE)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # GEMM chunk 和 Memory 操作交替
    def interleave_callback(i):
        _ = memory_operation(tensors, i)
    
    C = chunked_gemm_with_callback(A, B, interleave_callback, K_TILE)
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"  Total time: {total_time:.2f} ms")
    return total_time

def benchmark_two_streams():
    """两个 stream 并发：GEMM chunks 在 stream1，Memory 在 stream2"""
    print("\n" + "=" * 60)
    print("3. TWO STREAMS: GEMM chunks || Memory ops")
    print("=" * 60)
    
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    tensors = [torch.randn(2048, 2048, device='cuda') for _ in range(3)]
    torch.cuda.synchronize()
    
    stream_gemm = torch.cuda.Stream()
    stream_mem = torch.cuda.Stream()
    
    # Warmup
    with torch.cuda.stream(stream_gemm):
        _ = chunked_gemm_with_callback(A, B, k_tile=K_TILE)
    with torch.cuda.stream(stream_mem):
        for i in range(NUM_CHUNKS):
            _ = memory_operation(tensors, i)
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    # 同时提交到两个 stream
    with torch.cuda.stream(stream_gemm):
        C = chunked_gemm_with_callback(A, B, k_tile=K_TILE)
    
    with torch.cuda.stream(stream_mem):
        for i in range(NUM_CHUNKS):
            _ = memory_operation(tensors, i)
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"  Total time: {total_time:.2f} ms")
    return total_time

def benchmark_two_threads_streams():
    """两个线程+两个 stream：真正的并发启动"""
    print("\n" + "=" * 60)
    print("4. TWO THREADS + STREAMS: Concurrent kernel launch")
    print("=" * 60)
    
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    tensors = [torch.randn(2048, 2048, device='cuda') for _ in range(3)]
    torch.cuda.synchronize()
    
    stream_gemm = torch.cuda.Stream()
    stream_mem = torch.cuda.Stream()
    
    def gemm_worker():
        with torch.cuda.stream(stream_gemm):
            return chunked_gemm_with_callback(A, B, k_tile=K_TILE)
    
    def mem_worker():
        with torch.cuda.stream(stream_mem):
            for i in range(NUM_CHUNKS):
                _ = memory_operation(tensors, i)
    
    # Warmup
    t1 = threading.Thread(target=gemm_worker)
    t2 = threading.Thread(target=mem_worker)
    t1.start(); t2.start(); t1.join(); t2.join()
    torch.cuda.synchronize()
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    t1 = threading.Thread(target=gemm_worker)
    t2 = threading.Thread(target=mem_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) * 1000
    
    print(f"  Total time: {total_time:.2f} ms")
    return total_time

def main():
    print("=" * 60)
    print("EXPERIMENT: Chunked GEMM for Concurrent Scheduling")
    print("=" * 60)
    print(f"GEMM size: {M}x{K} @ {K}x{N}")
    print(f"K_TILE: {K_TILE}, NUM_CHUNKS: {NUM_CHUNKS}")
    print()
    
    time_seq = benchmark_sequential()
    time_interleaved = benchmark_interleaved()
    time_streams = benchmark_two_streams()
    time_threads = benchmark_two_threads_streams()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<35} {'Time':>10} {'Speedup':>10}")
    print("-" * 60)
    print(f"{'1. Sequential':<35} {time_seq:>8.1f} ms {'1.00x':>10}")
    print(f"{'2. Interleaved (same stream)':<35} {time_interleaved:>8.1f} ms {time_seq/time_interleaved:>9.2f}x")
    print(f"{'3. Two streams (single thread)':<35} {time_streams:>8.1f} ms {time_seq/time_streams:>9.2f}x")
    print(f"{'4. Two threads + streams':<35} {time_threads:>8.1f} ms {time_seq/time_threads:>9.2f}x")

if __name__ == "__main__":
    main()
