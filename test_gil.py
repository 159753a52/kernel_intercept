"""
测试：验证调度器线程不受 GIL 影响
"""
import torch
import time
import threading
import ctypes
import sys

def test_native_python_threads():
    """Python 原生线程（受 GIL 影响）"""
    print("=" * 50)
    print("Test 1: Native Python threads (GIL bound)")
    print("=" * 50)
    
    def cpu_work():
        # 纯 Python CPU 计算（受 GIL）
        total = 0
        for i in range(1000000):
            total += i
        return total
    
    start = time.perf_counter()
    t1 = threading.Thread(target=cpu_work)
    t2 = threading.Thread(target=cpu_work)
    t1.start(); t2.start()
    t1.join(); t2.join()
    elapsed = (time.perf_counter() - start) * 1000
    
    # 单线程时间
    start2 = time.perf_counter()
    cpu_work()
    cpu_work()
    single = (time.perf_counter() - start2) * 1000
    
    print(f"  Two threads: {elapsed:.1f} ms")
    print(f"  Sequential:  {single:.1f} ms")
    print(f"  Speedup: {single/elapsed:.2f}x (should be ~1.0 due to GIL)")

def test_cuda_threads():
    """CUDA 操作线程（C++ 层，不受 GIL）"""
    print("\n" + "=" * 50)
    print("Test 2: CUDA threads (C++ layer, no GIL)")
    print("=" * 50)
    
    A = torch.randn(2048, 2048, device='cuda')
    B = torch.randn(2048, 2048, device='cuda')
    torch.cuda.synchronize()
    
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()
    
    def gpu_work(stream, name):
        with torch.cuda.stream(stream):
            for _ in range(10):
                C = torch.mm(A, B)
    
    # Two threads
    start = time.perf_counter()
    t1 = threading.Thread(target=gpu_work, args=(stream1, "T1"))
    t2 = threading.Thread(target=gpu_work, args=(stream2, "T2"))
    t1.start(); t2.start()
    t1.join(); t2.join()
    torch.cuda.synchronize()
    two_threads = (time.perf_counter() - start) * 1000
    
    # Sequential
    torch.cuda.synchronize()
    start = time.perf_counter()
    gpu_work(stream1, "S1")
    torch.cuda.synchronize()
    gpu_work(stream2, "S2")
    torch.cuda.synchronize()
    sequential = (time.perf_counter() - start) * 1000
    
    print(f"  Two threads: {two_threads:.1f} ms")
    print(f"  Sequential:  {sequential:.1f} ms")
    print(f"  Speedup: {sequential/two_threads:.2f}x (can be >1.0, no GIL)")

if __name__ == "__main__":
    test_native_python_threads()
    test_cuda_threads()
