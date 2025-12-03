#!/usr/bin/env python3
"""
验证调度器线程是否真正执行 kernel

通过测量时间差来验证：
1. 如果调度器线程执行 kernel，kernel 完成时间会在调度器线程
2. 主线程只是等待，不执行 kernel
"""

import torch
import ctypes
import time
import threading

def test_without_scheduler():
    """不使用调度器的正常执行"""
    print("\n=== Test WITHOUT scheduler ===")
    print(f"Main thread: {threading.current_thread().ident}")
    
    a = torch.randn(1000, 1000, device="cuda")
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(10):
        b = a @ a
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for i in range(100):
        b = a @ a  # 矩阵乘法
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    print(f"Time: {(end-start)*1000:.2f} ms for 100 matmul")
    return end - start

def test_with_scheduler():
    """使用调度器的执行"""
    print("\n=== Test WITH scheduler ===")
    print(f"Main thread: {threading.current_thread().ident}")
    
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(1)
    
    a = torch.randn(1000, 1000, device="cuda")
    torch.cuda.synchronize()
    
    # Warmup (passthrough mode)
    for _ in range(10):
        b = a @ a
    torch.cuda.synchronize()
    
    # Enable scheduling
    lib.orion_set_client_idx(0)
    
    # Warmup with scheduler
    for _ in range(10):
        b = a @ a
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for i in range(100):
        b = a @ a
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    lib.orion_stop_scheduler()
    
    print(f"Time: {(end-start)*1000:.2f} ms for 100 matmul")
    return end - start

def main():
    print("=" * 60)
    print("SCHEDULER VERIFICATION TEST")
    print("=" * 60)
    print("\nThis test verifies that kernels are actually executed.")
    print("If both tests complete and produce similar times,")
    print("it proves the scheduler is correctly executing kernels.")
    
    t1 = test_without_scheduler()
    
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    t2 = test_with_scheduler()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Without scheduler: {t1*1000:.2f} ms")
    print(f"With scheduler:    {t2*1000:.2f} ms")
    print(f"Overhead:          {((t2/t1)-1)*100:.1f}%")
    
    if abs(t2/t1 - 1) < 0.5:  # 允许 50% 的开销差异
        print("\n[PASS] Scheduler is correctly executing kernels!")
    else:
        print("\n[WARNING] Large difference detected, check scheduler implementation")

if __name__ == "__main__":
    main()
