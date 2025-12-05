#!/usr/bin/env python3
"""
使用 PyTorch Profiler 对比：
1. 不分块：Full GEMM + Memory ops（顺序）
2. 分块：Chunked GEMM + Memory ops（并发）
"""

import torch
import torch.profiler
import threading
import os

M, N, K = 2048, 2048, 2048
K_TILE = 256
NUM_CHUNKS = K // K_TILE
NUM_MEMORY_OPS = 8

os.makedirs("profiles", exist_ok=True)

def setup_tensors():
    A = torch.randn(M, K, device='cuda')
    B = torch.randn(K, N, device='cuda')
    mem_tensors = [torch.randn(2048, 2048, device='cuda') for _ in range(3)]
    torch.cuda.synchronize()
    return A, B, mem_tensors

def full_gemm(A, B):
    return torch.mm(A, B)

def chunked_gemm(A, B, k_tile=K_TILE):
    M, K = A.shape
    _, N = B.shape
    C = None
    for k in range(0, K, k_tile):
        k_end = min(k + k_tile, K)
        if C is None:
            C = torch.mm(A[:, k:k_end], B[k:k_end, :])
        else:
            C = C + torch.mm(A[:, k:k_end], B[k:k_end, :])
    return C

def memory_ops(tensors, num_ops=NUM_MEMORY_OPS):
    for _ in range(num_ops):
        x = tensors[0] + tensors[1]
        x = x * tensors[2]
        x = torch.relu(x)
    return x

def profile_sequential_full():
    print("Profile 1: Sequential Full GEMM")
    A, B, mem_tensors = setup_tensors()
    _ = full_gemm(A, B); _ = memory_ops(mem_tensors); torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        C = full_gemm(A, B)
        _ = memory_ops(mem_tensors)
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/1_sequential_full_gemm.json")
    print("  Saved: profiles/1_sequential_full_gemm.json")

def profile_two_streams_full():
    print("Profile 2: Two streams Full GEMM")
    A, B, mem_tensors = setup_tensors()
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    with torch.cuda.stream(stream1): _ = full_gemm(A, B)
    with torch.cuda.stream(stream2): _ = memory_ops(mem_tensors)
    torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        def w1():
            with torch.cuda.stream(stream1): full_gemm(A, B)
        def w2():
            with torch.cuda.stream(stream2): memory_ops(mem_tensors)
        t1 = threading.Thread(target=w1); t2 = threading.Thread(target=w2)
        t1.start(); t2.start(); t1.join(); t2.join()
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/2_two_streams_full_gemm.json")
    print("  Saved: profiles/2_two_streams_full_gemm.json")

def profile_sequential_chunked():
    print("Profile 3: Sequential Chunked GEMM")
    A, B, mem_tensors = setup_tensors()
    _ = chunked_gemm(A, B); _ = memory_ops(mem_tensors); torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        C = chunked_gemm(A, B)
        _ = memory_ops(mem_tensors)
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/3_sequential_chunked_gemm.json")
    print("  Saved: profiles/3_sequential_chunked_gemm.json")

def profile_two_streams_chunked():
    print("Profile 4: Two streams Chunked GEMM")
    A, B, mem_tensors = setup_tensors()
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    with torch.cuda.stream(stream1): _ = chunked_gemm(A, B)
    with torch.cuda.stream(stream2): _ = memory_ops(mem_tensors)
    torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        def w1():
            with torch.cuda.stream(stream1): chunked_gemm(A, B)
        def w2():
            with torch.cuda.stream(stream2): memory_ops(mem_tensors)
        t1 = threading.Thread(target=w1); t2 = threading.Thread(target=w2)
        t1.start(); t2.start(); t1.join(); t2.join()
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/4_two_streams_chunked_gemm.json")
    print("  Saved: profiles/4_two_streams_chunked_gemm.json")

if __name__ == "__main__":
    print(f"GEMM: {M}x{K} @ {K}x{N}, K_TILE={K_TILE}, {NUM_CHUNKS} chunks")
    profile_sequential_full()
    profile_two_streams_full()
    profile_sequential_chunked()
    profile_two_streams_chunked()
    print("\nFiles in profiles/. Open with chrome://tracing")
