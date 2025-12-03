#!/usr/bin/env python3
"""
生成完整的 PyTorch Profiler 分析文件
包含 GPT 模型推理的完整过程
"""

import torch
import torch.profiler
from torch.profiler import ProfilerActivity
import ctypes
import os
import sys
import json

sys.path.insert(0, ".")
from GPT import create_char_gpt, sequence_len

def profile_normal():
    """正常执行的完整 profile"""
    print("=" * 70)
    print("PROFILING NORMAL EXECUTION (no scheduler)")
    print("=" * 70)
    
    # 创建模型
    print("Creating GPT model...")
    model = create_char_gpt(256, "cuda")
    model.eval()
    dummy_input = torch.randint(0, 256, (2, sequence_len), device="cuda")
    
    # Warmup
    print("Warmup...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # Profile
    print("Profiling 5 inference iterations...")
    with torch.profiler.profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for i in range(5):
            with torch.no_grad():
                output = model(dummy_input)
            torch.cuda.synchronize()
    
    # 导出
    prof.export_chrome_trace("profiles/full_normal.json")
    
    # 打印统计
    print("\n--- Top 15 CUDA Operations ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    # 保存文本报告
    with open("profiles/full_normal_stats.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("NORMAL EXECUTION PROFILE (No Scheduler)\n")
        f.write("=" * 80 + "\n\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    
    del model
    torch.cuda.empty_cache()
    print("\nSaved: profiles/full_normal.json, profiles/full_normal_stats.txt")

def profile_scheduled():
    """调度器执行的完整 profile"""
    print("\n" + "=" * 70)
    print("PROFILING SCHEDULED EXECUTION (with scheduler)")
    print("=" * 70)
    
    # 加载调度器
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(1)
    
    # 创建模型
    print("Creating GPT model...")
    model = create_char_gpt(256, "cuda")
    model.eval()
    dummy_input = torch.randint(0, 256, (2, sequence_len), device="cuda")
    
    # Warmup (passthrough mode)
    print("Warmup (passthrough)...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # Enable scheduling
    lib.orion_set_client_idx(0)
    print("Scheduler enabled for this thread")
    
    # Warmup with scheduler
    print("Warmup (scheduled)...")
    for _ in range(2):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # Profile
    print("Profiling 5 inference iterations...")
    with torch.profiler.profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for i in range(5):
            with torch.no_grad():
                output = model(dummy_input)
            torch.cuda.synchronize()
    
    # 导出
    prof.export_chrome_trace("profiles/full_scheduled.json")
    
    # 打印统计
    print("\n--- Top 15 CUDA Operations ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    # 保存文本报告
    with open("profiles/full_scheduled_stats.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("SCHEDULED EXECUTION PROFILE (With Scheduler)\n")
        f.write("=" * 80 + "\n\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    
    lib.orion_stop_scheduler()
    del model
    torch.cuda.empty_cache()
    print("\nSaved: profiles/full_scheduled.json, profiles/full_scheduled_stats.txt")

def analyze_threads():
    """分析两个 profile 文件中的线程分布"""
    print("\n" + "=" * 70)
    print("THREAD ANALYSIS")
    print("=" * 70)
    
    for name, filepath in [("Normal", "profiles/full_normal.json"), 
                           ("Scheduled", "profiles/full_scheduled.json")]:
        if not os.path.exists(filepath):
            continue
            
        with open(filepath) as f:
            data = json.load(f)
        
        # 统计线程
        threads = {}
        for event in data.get("traceEvents", []):
            tid = event.get("tid")
            cat = event.get("cat", "")
            if tid is not None:
                if tid not in threads:
                    threads[tid] = {"total": 0, "cuda_runtime": 0, "kernel": 0, "cpu_op": 0}
                threads[tid]["total"] += 1
                if cat == "cuda_runtime":
                    threads[tid]["cuda_runtime"] += 1
                elif cat == "kernel":
                    threads[tid]["kernel"] += 1
                elif cat == "cpu_op":
                    threads[tid]["cpu_op"] += 1
        
        print(f"\n{name} Execution - Threads with CUDA activity:")
        for tid, stats in sorted(threads.items(), key=lambda x: -x[1]["cuda_runtime"]):
            if stats["cuda_runtime"] > 0:
                print(f"  Thread {tid}:")
                print(f"    CUDA Runtime calls: {stats['cuda_runtime']}")
                print(f"    Total events: {stats['total']}")

def main():
    os.makedirs("profiles", exist_ok=True)
    
    # 检查是否使用 LD_PRELOAD
    use_scheduler = "libgpu_scheduler.so" in os.environ.get("LD_PRELOAD", "")
    
    if use_scheduler:
        profile_scheduled()
    else:
        profile_normal()
    
    # 分析线程
    analyze_threads()
    
    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    for f in os.listdir("profiles"):
        if f.startswith("full_"):
            fpath = os.path.join("profiles", f)
            size = os.path.getsize(fpath)
            print(f"  {f}: {size/1024/1024:.1f} MB" if size > 1024*1024 else f"  {f}: {size/1024:.1f} KB")
    
    print("\nTo view traces in Chrome:")
    print("  1. Open chrome://tracing")
    print("  2. Load the JSON files")

if __name__ == "__main__":
    main()
