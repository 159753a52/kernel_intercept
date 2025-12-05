#!/usr/bin/env python3
"""
分块 GEMM 实现：把大 GEMM 沿 K 维拆分，给调度器创造并发机会

数学原理：
  C = α * A @ B + β * C
  
  拆分为：
  C = α * Σ(A[:, Ki] @ B[Ki, :]) + β * C
  
  第一块：C = α * A[:, 0:K0] @ B[0:K0, :] + β * C
  第二块：C = α * A[:, K0:K1] @ B[K0:K1, :] + 1.0 * C  (β=1 累加)
  ...
"""

import torch
import time

def chunked_gemm(A, B, C=None, alpha=1.0, beta=0.0, k_tile=256):
    """
    分块 GEMM：C = alpha * A @ B + beta * C
    
    Args:
        A: (M, K) 矩阵
        B: (K, N) 矩阵
        C: (M, N) 输出矩阵 (可选)
        alpha, beta: 标量
        k_tile: K 维分块大小
    
    Returns:
        C: 结果矩阵
    """
    M, K = A.shape
    _, N = B.shape
    
    if C is None:
        C = torch.zeros(M, N, device=A.device, dtype=A.dtype)
    
    # 沿 K 维分块
    for k in range(0, K, k_tile):
        k_end = min(k + k_tile, K)
        
        # 取子矩阵
        A_chunk = A[:, k:k_end]  # (M, k_tile)
        B_chunk = B[k:k_end, :]  # (k_tile, N)
        
        # 计算部分结果
        if k == 0:
            # 第一块：使用原始 beta
            C = alpha * torch.mm(A_chunk, B_chunk) + beta * C
        else:
            # 后续块：累加到 C (beta=1)
            C = C + alpha * torch.mm(A_chunk, B_chunk)
    
    return C


def benchmark_chunked_vs_full():
    """对比分块 GEMM 和完整 GEMM 的性能"""
    print("=" * 60)
    print("Chunked GEMM vs Full GEMM Benchmark")
    print("=" * 60)
    
    M, N, K = 4096, 4096, 4096
    
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    C = torch.zeros(M, N, device='cuda')
    torch.cuda.synchronize()
    
    # Warmup
    _ = torch.mm(A, B)
    _ = chunked_gemm(A, B, k_tile=512)
    torch.cuda.synchronize()
    
    # Full GEMM
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        C_full = torch.mm(A, B)
    torch.cuda.synchronize()
    full_time = (time.perf_counter() - start) * 1000 / 10
    
    print(f"\nFull GEMM ({M}x{K} @ {K}x{N}):")
    print(f"  Time: {full_time:.2f} ms")
    
    # Chunked GEMM with different k_tile
    for k_tile in [256, 512, 1024, 2048]:
        num_chunks = (K + k_tile - 1) // k_tile
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            C_chunked = chunked_gemm(A, B, k_tile=k_tile)
        torch.cuda.synchronize()
        chunked_time = (time.perf_counter() - start) * 1000 / 10
        
        overhead = (chunked_time / full_time - 1) * 100
        print(f"\nChunked GEMM (k_tile={k_tile}, {num_chunks} chunks):")
        print(f"  Time: {chunked_time:.2f} ms")
        print(f"  Overhead: {overhead:+.1f}%")
        
        # 验证正确性
        if not torch.allclose(C_full, C_chunked, rtol=1e-3, atol=1e-3):
            print("  WARNING: Results differ!")
        else:
            print("  Correctness: OK")


def estimate_chunk_time():
    """估算不同 k_tile 的单个 chunk 执行时间"""
    print("\n" + "=" * 60)
    print("Estimating per-chunk execution time")
    print("=" * 60)
    
    M, N = 4096, 4096
    
    for k_tile in [128, 256, 512, 1024]:
        A = torch.randn(M, k_tile, device='cuda')
        B = torch.randn(k_tile, N, device='cuda')
        torch.cuda.synchronize()
        
        # Warmup
        for _ in range(5):
            _ = torch.mm(A, B)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = torch.mm(A, B)
        torch.cuda.synchronize()
        chunk_time = (time.perf_counter() - start) * 1000 / 100
        
        print(f"k_tile={k_tile:4d}: {chunk_time:.3f} ms per chunk")


if __name__ == "__main__":
    benchmark_chunked_vs_full()
    estimate_chunk_time()
