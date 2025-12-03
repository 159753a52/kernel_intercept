#!/usr/bin/env python3
"""
使用 PyTorch Profiler 对比正常运行和调度拦截执行的性能
"""

import torch
import torch.profiler
import sys
import os
import ctypes

sys.path.insert(0, ".")
from GPT import create_char_gpt, sequence_len

def run_inference(model, dummy_input, num_iters=5):
    """运行推理"""
    for _ in range(num_iters):
        with torch.no_grad():
            output = model(dummy_input)
        torch.cuda.synchronize()
    return output

def profile_normal():
    """正常运行的 profile"""
    print("=" * 60)
    print("Profiling NORMAL execution (no scheduler)")
    print("=" * 60)
    
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
    print("Profiling...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        run_inference(model, dummy_input, num_iters=3)
    
    # 导出 Chrome trace
    prof.export_chrome_trace("profiles/normal_trace.json")
    
    # 打印统计
    print("\nTop 20 CUDA operations:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # 保存文本报告
    with open("profiles/normal_stats.txt", "w") as f:
        f.write("Normal Execution Profile\n")
        f.write("=" * 80 + "\n\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    
    print("\nSaved: profiles/normal_trace.json, profiles/normal_stats.txt")
    
    del model
    torch.cuda.empty_cache()

def profile_scheduled():
    """调度拦截执行的 profile"""
    print("\n" + "=" * 60)
    print("Profiling SCHEDULED execution (with interceptor)")
    print("=" * 60)
    
    # 加载调度器库
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(1)
    
    model = create_char_gpt(256, "cuda")
    model.eval()
    dummy_input = torch.randint(0, 256, (2, sequence_len), device="cuda")
    
    # Warmup (passthrough mode)
    print("Warmup (passthrough)...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # 启用调度
    lib.orion_set_client_idx(0)
    print("Scheduler enabled for this thread")
    
    # Profile
    print("Profiling...")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        run_inference(model, dummy_input, num_iters=3)
    
    # 导出 Chrome trace
    prof.export_chrome_trace("profiles/scheduled_trace.json")
    
    # 打印统计
    print("\nTop 20 CUDA operations:")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    
    # 保存文本报告
    with open("profiles/scheduled_stats.txt", "w") as f:
        f.write("Scheduled Execution Profile\n")
        f.write("=" * 80 + "\n\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    
    print("\nSaved: profiles/scheduled_trace.json, profiles/scheduled_stats.txt")
    
    lib.orion_stop_scheduler()
    del model
    torch.cuda.empty_cache()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["normal", "scheduled", "both"], default="both")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs("profiles", exist_ok=True)
    
    if args.mode in ["normal", "both"]:
        profile_normal()
    
    if args.mode in ["scheduled", "both"]:
        profile_scheduled()
    
    print("\n" + "=" * 60)
    print("PROFILING COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    if args.mode in ["normal", "both"]:
        print("  - profiles/normal_trace.json     (Chrome trace, normal)")
        print("  - profiles/normal_stats.txt      (Text stats, normal)")
    if args.mode in ["scheduled", "both"]:
        print("  - profiles/scheduled_trace.json  (Chrome trace, scheduled)")
        print("  - profiles/scheduled_stats.txt   (Text stats, scheduled)")
    print("\nTo view traces: Open chrome://tracing and load the JSON files")

if __name__ == "__main__":
    main()
