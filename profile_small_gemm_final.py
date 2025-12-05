#!/usr/bin/env python3
"""
实验：小 GEMM (不占满 GPU) + Memory 操作并发

配置：
- GEMM: 512x512 (~16 blocks, A30 有 56 SM，不占满)
- Memory: 2048x2048 张量操作
- 迭代: 10 次

预期结果：小 GEMM 不占满 GPU，可以和 Memory 操作并发执行
"""

import torch
import torch.profiler
import os

GEMM_SIZE = 512
MEM_SIZE = 2048
NUM_ITERS = 10

os.makedirs("profiles", exist_ok=True)

def setup():
    A = torch.randn(GEMM_SIZE, GEMM_SIZE, device='cuda')
    B = torch.randn(GEMM_SIZE, GEMM_SIZE, device='cuda')
    X = torch.randn(MEM_SIZE, MEM_SIZE, device='cuda')
    Y = torch.randn(MEM_SIZE, MEM_SIZE, device='cuda')
    Z = torch.randn(MEM_SIZE, MEM_SIZE, device='cuda')
    torch.cuda.synchronize()
    return A, B, X, Y, Z

def profile_sequential():
    """顺序执行"""
    print("Profile 1: 顺序执行")
    A, B, X, Y, Z = setup()
    
    # Warmup
    for _ in range(20):
        for _ in range(NUM_ITERS):
            _ = torch.mm(A, B)
            _ = X + Y
        torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(NUM_ITERS):
            _ = torch.mm(A, B)
            r = X + Y
            r = r * Z
            r = torch.relu(r)
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/small_gemm_final_1_sequential.json")
    print("  Saved: profiles/small_gemm_final_1_sequential.json")

def profile_concurrent():
    """并发执行"""
    print("Profile 2: 并发执行")
    A, B, X, Y, Z = setup()
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # Warmup
    for _ in range(20):
        with torch.cuda.stream(stream1):
            for _ in range(NUM_ITERS):
                _ = torch.mm(A, B)
        with torch.cuda.stream(stream2):
            for _ in range(NUM_ITERS):
                _ = X + Y
        torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        with torch.cuda.stream(stream1):
            for _ in range(NUM_ITERS):
                _ = torch.mm(A, B)
        with torch.cuda.stream(stream2):
            for _ in range(NUM_ITERS):
                r = X + Y
                r = r * Z
                r = torch.relu(r)
        stream1.synchronize()
        stream2.synchronize()
    
    prof.export_chrome_trace("profiles/small_gemm_final_2_concurrent.json")
    print("  Saved: profiles/small_gemm_final_2_concurrent.json")

def measure():
    """精确测量"""
    print("\n" + "="*60)
    print("精确测量 (CUDA Event)")
    print("="*60)
    
    A, B, X, Y, Z = setup()
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # Warmup
    for _ in range(20):
        for _ in range(NUM_ITERS):
            _ = torch.mm(A, B)
            _ = X + Y
        torch.cuda.synchronize()
    
    # 顺序
    times = []
    for _ in range(20):
        start.record()
        for _ in range(NUM_ITERS):
            _ = torch.mm(A, B)
            r = X + Y
            r = r * Z
            r = torch.relu(r)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    seq = sum(times[5:]) / len(times[5:])
    
    # Warmup 并发
    for _ in range(20):
        with torch.cuda.stream(stream1):
            for _ in range(NUM_ITERS): _ = torch.mm(A, B)
        with torch.cuda.stream(stream2):
            for _ in range(NUM_ITERS): _ = X + Y
        torch.cuda.synchronize()
    
    # 并发
    times = []
    for _ in range(20):
        start.record()
        with torch.cuda.stream(stream1):
            for _ in range(NUM_ITERS):
                _ = torch.mm(A, B)
        with torch.cuda.stream(stream2):
            for _ in range(NUM_ITERS):
                r = X + Y
                r = r * Z
                r = torch.relu(r)
        stream1.synchronize()
        stream2.synchronize()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    conc = sum(times[5:]) / len(times[5:])
    
    print(f"顺序: {seq:.2f} ms")
    print(f"并发: {conc:.2f} ms")
    print(f"加速: {seq/conc:.2f}x")
    
    return seq, conc

if __name__ == "__main__":
    print("="*60)
    print("小 GEMM + Memory 并发实验")
    print("="*60)
    print(f"GEMM: {GEMM_SIZE}x{GEMM_SIZE} (~16 blocks, 不占满 56 SM)")
    print(f"Memory: {MEM_SIZE}x{MEM_SIZE}")
    print(f"迭代: {NUM_ITERS} 次")
    print()
    
    profile_sequential()
    profile_concurrent()
    seq, conc = measure()
    
    print("\n" + "="*60)
    print("Profile 文件")
    print("="*60)
    print("  profiles/small_gemm_final_1_sequential.json - 顺序")
    print("  profiles/small_gemm_final_2_concurrent.json - 并发")
    print("\n打开: chrome://tracing")
