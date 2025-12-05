#!/usr/bin/env python3
"""
生成 PyTorch Profiler 对比文件：
1. 无 LD_PRELOAD，顺序执行
2. 无 LD_PRELOAD，两线程并发
3. 有调度器，两客户端并发
"""

import torch
import torch.profiler
import threading
import os
import ctypes
import sys

SIZE = 2048
NUM_OPS = 10

os.makedirs("profiles", exist_ok=True)

def setup():
    A = torch.randn(SIZE, SIZE, device='cuda')
    B = torch.randn(SIZE, SIZE, device='cuda')
    X = torch.randn(SIZE, SIZE, device='cuda')
    torch.cuda.synchronize()
    return A, B, X

def profile_sequential():
    """顺序执行"""
    print("Profile 1: Sequential (no preload)")
    A, B, X = setup()
    
    # Warmup
    for _ in range(3):
        _ = torch.mm(A, B)
        _ = X + X
    torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, 
                   torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        for _ in range(NUM_OPS):
            C = torch.mm(A, B)
        for _ in range(NUM_OPS):
            Y = X + X
            Y = Y * X
            Y = torch.relu(Y)
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/scheduler_1_sequential.json")
    print("  Saved: profiles/scheduler_1_sequential.json")

def profile_two_threads():
    """两线程并发"""
    print("Profile 2: Two threads + streams (no preload)")
    A, B, X = setup()
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    # Warmup
    def w1():
        with torch.cuda.stream(stream1):
            for _ in range(NUM_OPS):
                _ = torch.mm(A, B)
    def w2():
        with torch.cuda.stream(stream2):
            for _ in range(NUM_OPS):
                _ = X + X
    t1 = threading.Thread(target=w1)
    t2 = threading.Thread(target=w2)
    t1.start(); t2.start(); t1.join(); t2.join()
    torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        def w1():
            with torch.cuda.stream(stream1):
                for _ in range(NUM_OPS):
                    C = torch.mm(A, B)
        def w2():
            with torch.cuda.stream(stream2):
                for _ in range(NUM_OPS):
                    Y = X + X
                    Y = Y * X
                    Y = torch.relu(Y)
        t1 = threading.Thread(target=w1)
        t2 = threading.Thread(target=w2)
        t1.start(); t2.start(); t1.join(); t2.join()
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/scheduler_2_two_threads.json")
    print("  Saved: profiles/scheduler_2_two_threads.json")

def profile_with_scheduler():
    """使用调度器"""
    print("Profile 3: With scheduler (LD_PRELOAD)")
    
    # 检查是否有 LD_PRELOAD
    if "libgpu_scheduler.so" not in os.environ.get("LD_PRELOAD", ""):
        print("  [SKIP] Run with LD_PRELOAD=./build/libgpu_scheduler.so")
        return
    
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(2)
    
    A, B, X = setup()
    
    def client0():
        lib.orion_set_client_idx(0)
        for _ in range(NUM_OPS):
            C = torch.mm(A, B)
        torch.cuda.synchronize()
    
    def client1():
        lib.orion_set_client_idx(1)
        for _ in range(NUM_OPS):
            Y = X + X
            Y = Y * X
            Y = torch.relu(Y)
        torch.cuda.synchronize()
    
    # Warmup
    t1 = threading.Thread(target=client0)
    t2 = threading.Thread(target=client1)
    t1.start(); t2.start(); t1.join(); t2.join()
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                   torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        t1 = threading.Thread(target=client0)
        t2 = threading.Thread(target=client1)
        t1.start(); t2.start(); t1.join(); t2.join()
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/scheduler_3_with_scheduler.json")
    print("  Saved: profiles/scheduler_3_with_scheduler.json")
    
    lib.orion_stop_scheduler()

if __name__ == "__main__":
    print(f"Size: {SIZE}x{SIZE}, Ops: {NUM_OPS}")
    profile_sequential()
    profile_two_threads()
    profile_with_scheduler()
    print("\nOpen with chrome://tracing")
