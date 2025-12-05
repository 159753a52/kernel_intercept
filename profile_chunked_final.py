#!/usr/bin/env python3
"""
最终实验：分块 GEMM 并发分析

结论：在 A30 GPU 上，分块并发没有带来加速，原因：
1. GEMM chunk 仍然占用大量 GPU 资源
2. 分块带来额外开销（更多 kernel launch，额外内存读写）
"""

import torch
import torch.profiler
import os

M, N, K = 4096, 4096, 4096
K_TILE = 512
NUM_CHUNKS = K // K_TILE

os.makedirs("profiles", exist_ok=True)

def setup():
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    mem_tensors = [torch.randn(4096, 4096, device='cuda') for _ in range(3)]
    torch.cuda.synchronize()
    return A, B, mem_tensors

def memory_op(tensors):
    x = tensors[0] + tensors[1]
    x = x * tensors[2]
    return torch.relu(x)

def profile_no_chunking():
    """不分块：完整 GEMM → Memory ops"""
    print("Profile: 不分块")
    A, B, mem_tensors = setup()
    
    # Warmup
    for _ in range(10):
        _ = torch.mm(A, B)
        for _ in range(NUM_CHUNKS):
            _ = memory_op(mem_tensors)
        torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        C = torch.mm(A, B)
        for _ in range(NUM_CHUNKS):
            _ = memory_op(mem_tensors)
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/final_1_no_chunking.json")
    print("  Saved: profiles/final_1_no_chunking.json")

def profile_chunked_concurrent():
    """分块并发"""
    print("Profile: 分块并发")
    A, B, mem_tensors = setup()
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # Warmup
    for _ in range(10):
        C = torch.zeros(M, N, device='cuda')
        for i in range(NUM_CHUNKS):
            k = i * K_TILE
            k_end = k + K_TILE
            with torch.cuda.stream(stream1):
                if i == 0:
                    C = torch.mm(A[:, k:k_end], B[k:k_end, :])
                else:
                    C = C + torch.mm(A[:, k:k_end], B[k:k_end, :])
            with torch.cuda.stream(stream2):
                _ = memory_op(mem_tensors)
        stream1.synchronize()
        stream2.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        C = torch.zeros(M, N, device='cuda')
        for i in range(NUM_CHUNKS):
            k = i * K_TILE
            k_end = k + K_TILE
            with torch.cuda.stream(stream1):
                if i == 0:
                    C = torch.mm(A[:, k:k_end], B[k:k_end, :])
                else:
                    C = C + torch.mm(A[:, k:k_end], B[k:k_end, :])
            with torch.cuda.stream(stream2):
                _ = memory_op(mem_tensors)
        stream1.synchronize()
        stream2.synchronize()
    
    prof.export_chrome_trace("profiles/final_2_chunked_concurrent.json")
    print("  Saved: profiles/final_2_chunked_concurrent.json")

def measure_times():
    """精确测量时间"""
    print("\n" + "="*60)
    print("精确测量 (CUDA Event, 20 runs)")
    print("="*60)
    
    A, B, mem_tensors = setup()
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(10):
        _ = torch.mm(A, B)
        for _ in range(NUM_CHUNKS):
            _ = memory_op(mem_tensors)
        torch.cuda.synchronize()
    
    # 顺序执行
    times = []
    for _ in range(20):
        start.record()
        C = torch.mm(A, B)
        for _ in range(NUM_CHUNKS):
            _ = memory_op(mem_tensors)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    avg_seq = sum(times) / len(times)
    print(f"顺序执行: {avg_seq:.2f} ms")
    
    # 分块并发
    for _ in range(10):
        C = torch.zeros(M, N, device='cuda')
        for i in range(NUM_CHUNKS):
            k = i * K_TILE
            k_end = k + K_TILE
            with torch.cuda.stream(stream1):
                C = C + torch.mm(A[:, k:k_end], B[k:k_end, :]) if i > 0 else torch.mm(A[:, k:k_end], B[k:k_end, :])
            with torch.cuda.stream(stream2):
                _ = memory_op(mem_tensors)
        stream1.synchronize()
        stream2.synchronize()
    
    times = []
    for _ in range(20):
        start.record()
        C = torch.zeros(M, N, device='cuda')
        for i in range(NUM_CHUNKS):
            k = i * K_TILE
            k_end = k + K_TILE
            with torch.cuda.stream(stream1):
                if i == 0:
                    C = torch.mm(A[:, k:k_end], B[k:k_end, :])
                else:
                    C = C + torch.mm(A[:, k:k_end], B[k:k_end, :])
            with torch.cuda.stream(stream2):
                _ = memory_op(mem_tensors)
        stream1.synchronize()
        stream2.synchronize()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    avg_conc = sum(times) / len(times)
    print(f"分块并发: {avg_conc:.2f} ms")
    print(f"加速比: {avg_seq/avg_conc:.2f}x")
    
    return avg_seq, avg_conc

if __name__ == "__main__":
    print("="*60)
    print("分块 GEMM 并发实验 - 最终版")
    print("="*60)
    print(f"GEMM: {M}x{K} @ {K}x{N}")
    print(f"K_TILE: {K_TILE}, NUM_CHUNKS: {NUM_CHUNKS}")
    print()
    
    profile_no_chunking()
    profile_chunked_concurrent()
    
    avg_seq, avg_conc = measure_times()
    
    print("\n" + "="*60)
    print("结论")
    print("="*60)
    if avg_conc < avg_seq:
        print(f"分块并发有效！加速 {avg_seq/avg_conc:.2f}x")
    else:
        print(f"分块并发无效！反而慢了 {avg_conc/avg_seq:.2f}x")
        print("\n可能原因:")
        print("1. GEMM chunk (~2.8ms) 仍然占满 GPU，无法并发")
        print("2. 分块带来额外开销 (kernel launch, 内存读写)")
        print("3. A30 GPU 的 compute/memory 并发能力有限")
    
    print("\nProfile 文件:")
    print("  profiles/final_1_no_chunking.json      - 不分块")
    print("  profiles/final_2_chunked_concurrent.json - 分块并发")
    print("\n打开: chrome://tracing")
