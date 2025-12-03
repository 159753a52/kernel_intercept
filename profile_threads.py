#!/usr/bin/env python3
"""
生成能看到客户端线程和调度器线程的 profile
使用 LD_PRELOAD 运行此脚本
"""

import torch
import torch.profiler
from torch.profiler import ProfilerActivity
import ctypes
import os
import json
import sys

def profile_normal():
    """正常执行（不使用调度器）的 profile"""
    print("\n" + "=" * 60)
    print("NORMAL EXECUTION (no scheduler)")
    print("=" * 60)
    
    a = torch.randn(100, 100, device="cuda")
    torch.cuda.synchronize()
    
    # Warmup
    for _ in range(5):
        b = a @ a
    torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(20):
            b = a @ a
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/normal_threads.json")
    return analyze_trace("profiles/normal_threads.json")

def profile_scheduled():
    """调度器执行的 profile"""
    print("\n" + "=" * 60)
    print("SCHEDULED EXECUTION (with scheduler)")
    print("=" * 60)
    
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(1)
    
    a = torch.randn(100, 100, device="cuda")
    torch.cuda.synchronize()
    
    # Warmup (passthrough)
    for _ in range(5):
        b = a @ a
    torch.cuda.synchronize()
    
    # Enable scheduling
    lib.orion_set_client_idx(0)
    
    # Warmup with scheduling
    for _ in range(5):
        b = a @ a
    torch.cuda.synchronize()
    
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(20):
            b = a @ a
        torch.cuda.synchronize()
    
    prof.export_chrome_trace("profiles/scheduled_threads.json")
    
    lib.orion_stop_scheduler()
    return analyze_trace("profiles/scheduled_threads.json")

def analyze_trace(filepath):
    """分析 trace 文件"""
    with open(filepath, "r") as f:
        data = json.load(f)
    
    result = {
        "cuda_by_thread": {},
        "main_thread": None,
        "other_threads": []
    }
    
    for event in data.get("traceEvents", []):
        if event.get("cat") == "cuda_runtime":
            tid = event.get("tid")
            name = event.get("name")
            if tid not in result["cuda_by_thread"]:
                result["cuda_by_thread"][tid] = {}
            result["cuda_by_thread"][tid][name] = result["cuda_by_thread"][tid].get(name, 0) + 1
    
    # 识别主线程（通常有最多调用）
    if result["cuda_by_thread"]:
        main_tid = max(result["cuda_by_thread"].keys(), 
                      key=lambda t: sum(result["cuda_by_thread"][t].values()))
        result["main_thread"] = main_tid
        result["other_threads"] = [t for t in result["cuda_by_thread"].keys() if t != main_tid]
    
    return result

def main():
    os.makedirs("profiles", exist_ok=True)
    
    # 检查是否使用 LD_PRELOAD
    use_scheduler = "libgpu_scheduler.so" in os.environ.get("LD_PRELOAD", "")
    
    if use_scheduler:
        scheduled_result = profile_scheduled()
        
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        
        print("\nCUDA Runtime API calls by thread:")
        for tid, calls in scheduled_result["cuda_by_thread"].items():
            if tid == scheduled_result["main_thread"]:
                thread_name = "CLIENT THREAD (main)"
            else:
                thread_name = "SCHEDULER THREAD"
            print(f"\n  {thread_name} (tid={tid}):")
            for name, count in calls.items():
                print(f"    {name}: {count}")
        
        if scheduled_result["other_threads"]:
            print("\n" + "=" * 60)
            print("VERIFICATION: Scheduler thread detected!")
            print("=" * 60)
            print(f"  Main thread: {scheduled_result['main_thread']}")
            print(f"  Scheduler thread(s): {scheduled_result['other_threads']}")
            print("\n  The cudaDeviceSynchronize calls on the scheduler thread")
            print("  prove that the scheduler is executing operations.")
        
        print("\n  Files generated:")
        print("    - profiles/scheduled_threads.json")
    else:
        normal_result = profile_normal()
        
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS (Normal execution)")
        print("=" * 60)
        
        print("\nCUDA Runtime API calls by thread:")
        for tid, calls in normal_result["cuda_by_thread"].items():
            print(f"\n  Thread {tid}:")
            for name, count in calls.items():
                print(f"    {name}: {count}")
        
        print("\n  Files generated:")
        print("    - profiles/normal_threads.json")
        
        print("\n  NOTE: Run with LD_PRELOAD to see scheduler threads:")
        print("    LD_PRELOAD=./build/libgpu_scheduler.so python3 profile_threads.py")

if __name__ == "__main__":
    main()
