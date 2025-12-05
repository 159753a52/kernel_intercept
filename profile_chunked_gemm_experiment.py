#!/usr/bin/env python3
"""
实验：单个 GEMM 拆分成多块，与带宽受限算子交替执行

对比：
1. 不分块：完整 GEMM → Memory 操作（顺序）
2. 分块：GEMM chunk1 → Mem op1 → GEMM chunk2 → Mem op2 → ...（交替，两个 stream 并发）

目标：验证分块 GEMM 能否让 compute 和 memory 操作并发执行
"""

import torch
import torch.profiler
import time
import os

# 参数
M, N, K = 4096, 4096, 4096  # 大矩阵
K_TILE = 512                # 分块大小
NUM_CHUNKS = K // K_TILE    # 8 个 chunks
NUM_MEM_OPS = NUM_CHUNKS    # 每个 chunk 对应一个 memory 操作

os.makedirs("profiles", exist_ok=True)

def setup():
    """创建测试数据"""
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    # Memory 操作的张量（大矩阵，带宽受限）
    mem_tensors = [torch.randn(4096, 4096, device='cuda') for _ in range(3)]
    torch.cuda.synchronize()
    return A, B, mem_tensors

def memory_op(tensors):
    """单个带宽受限操作"""
    x = tensors[0] + tensors[1]
    x = x * tensors[2]
    x = torch.relu(x)
    return x

# ============================================================
# 实验 1: 不分块，顺序执行
# ============================================================
def experiment_1_no_chunking():
    """完整 GEMM，然后执行所有 Memory 操作"""
    print("=" * 60)
    print("实验 1: 不分块 (Full GEMM → All Memory ops)")
    print("=" * 60)
    
    A, B, mem_tensors = setup()
    
    # Warmup
    C = torch.mm(A, B)
    for _ in range(NUM_MEM_OPS):
        _ = memory_op(mem_tensors)
    torch.cuda.synchronize()
    
    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        # 完整 GEMM
        C = torch.mm(A, B)
        # 所有 Memory 操作
        for _ in range(NUM_MEM_OPS):
            _ = memory_op(mem_tensors)
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/chunked_exp_1_no_chunking.json")
    
    # 计时
    torch.cuda.synchronize()
    start = time.perf_counter()
    C = torch.mm(A, B)
    for _ in range(NUM_MEM_OPS):
        _ = memory_op(mem_tensors)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"  Time: {elapsed:.2f} ms")
    print(f"  Saved: profiles/chunked_exp_1_no_chunking.json")
    return elapsed

# ============================================================
# 实验 2: 分块，同一个 stream，顺序执行
# ============================================================
def experiment_2_chunked_sequential():
    """分块 GEMM，与 Memory 操作交替，但在同一个 stream（无并发）"""
    print("\n" + "=" * 60)
    print("实验 2: 分块但顺序 (GEMM chunk → Mem op → repeat, same stream)")
    print("=" * 60)
    
    A, B, mem_tensors = setup()
    
    # Warmup
    C = torch.zeros(M, N, device='cuda')
    for k in range(0, K, K_TILE):
        k_end = min(k + K_TILE, K)
        C = C + torch.mm(A[:, k:k_end], B[k:k_end, :])
        _ = memory_op(mem_tensors)
    torch.cuda.synchronize()
    
    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        C = torch.zeros(M, N, device='cuda')
        for k in range(0, K, K_TILE):
            k_end = min(k + K_TILE, K)
            # GEMM chunk
            if k == 0:
                C = torch.mm(A[:, k:k_end], B[k:k_end, :])
            else:
                C = C + torch.mm(A[:, k:k_end], B[k:k_end, :])
            # Memory 操作
            _ = memory_op(mem_tensors)
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/chunked_exp_2_chunked_sequential.json")
    
    # 计时
    torch.cuda.synchronize()
    start = time.perf_counter()
    C = torch.zeros(M, N, device='cuda')
    for k in range(0, K, K_TILE):
        k_end = min(k + K_TILE, K)
        if k == 0:
            C = torch.mm(A[:, k:k_end], B[k:k_end, :])
        else:
            C = C + torch.mm(A[:, k:k_end], B[k:k_end, :])
        _ = memory_op(mem_tensors)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"  Time: {elapsed:.2f} ms")
    print(f"  Saved: profiles/chunked_exp_2_chunked_sequential.json")
    return elapsed

# ============================================================
# 实验 3: 分块，两个 stream，并发执行
# ============================================================
def experiment_3_chunked_concurrent():
    """分块 GEMM 在 stream1，Memory 操作在 stream2，交替提交实现并发"""
    print("\n" + "=" * 60)
    print("实验 3: 分块并发 (GEMM chunks on stream1 || Mem ops on stream2)")
    print("=" * 60)
    
    A, B, mem_tensors = setup()
    
    stream_compute = torch.cuda.Stream()
    stream_memory = torch.cuda.Stream()
    
    # Warmup
    with torch.cuda.stream(stream_compute):
        C = torch.zeros(M, N, device='cuda')
        for k in range(0, K, K_TILE):
            C = C + torch.mm(A[:, k:min(k+K_TILE, K)], B[k:min(k+K_TILE, K), :])
    with torch.cuda.stream(stream_memory):
        for _ in range(NUM_MEM_OPS):
            _ = memory_op(mem_tensors)
    torch.cuda.synchronize()
    
    # Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        C = torch.zeros(M, N, device='cuda')
        
        for i in range(NUM_CHUNKS):
            k = i * K_TILE
            k_end = min(k + K_TILE, K)
            
            # GEMM chunk 在 compute stream
            with torch.cuda.stream(stream_compute):
                if i == 0:
                    C = torch.mm(A[:, k:k_end], B[k:k_end, :])
                else:
                    C = C + torch.mm(A[:, k:k_end], B[k:k_end, :])
            
            # Memory 操作在 memory stream
            with torch.cuda.stream(stream_memory):
                _ = memory_op(mem_tensors)
        
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/chunked_exp_3_chunked_concurrent.json")
    
    # 计时
    torch.cuda.synchronize()
    start = time.perf_counter()
    C = torch.zeros(M, N, device='cuda')
    for i in range(NUM_CHUNKS):
        k = i * K_TILE
        k_end = min(k + K_TILE, K)
        with torch.cuda.stream(stream_compute):
            if i == 0:
                C = torch.mm(A[:, k:k_end], B[k:k_end, :])
            else:
                C = C + torch.mm(A[:, k:k_end], B[k:k_end, :])
        with torch.cuda.stream(stream_memory):
            _ = memory_op(mem_tensors)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"  Time: {elapsed:.2f} ms")
    print(f"  Saved: profiles/chunked_exp_3_chunked_concurrent.json")
    return elapsed

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("分块 GEMM 并发实验")
    print("=" * 60)
    print(f"GEMM: {M}x{K} @ {K}x{N}")
    print(f"K_TILE: {K_TILE}, NUM_CHUNKS: {NUM_CHUNKS}")
    print(f"Memory tensor: 4096x4096, {NUM_MEM_OPS} ops")
    print()
    
    time1 = experiment_1_no_chunking()
    time2 = experiment_2_chunked_sequential()
    time3 = experiment_3_chunked_concurrent()
    
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    print(f"{'方案':<40} {'时间':>10} {'加速比':>10}")
    print("-" * 60)
    print(f"{'1. 不分块 (Full GEMM → Memory)':<40} {time1:>8.2f} ms {'1.00x':>10}")
    print(f"{'2. 分块顺序 (chunk → mem → repeat)':<40} {time2:>8.2f} ms {time1/time2:>9.2f}x")
    print(f"{'3. 分块并发 (chunks || mem ops)':<40} {time3:>8.2f} ms {time1/time3:>9.2f}x")
    print()
    print("Profile 文件:")
    print("  profiles/chunked_exp_1_no_chunking.json      - 不分块")
    print("  profiles/chunked_exp_2_chunked_sequential.json - 分块顺序")
    print("  profiles/chunked_exp_3_chunked_concurrent.json - 分块并发")
    print()
    print("打开方式: chrome://tracing 或 edge://tracing")

if __name__ == "__main__":
    main()
