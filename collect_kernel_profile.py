#!/usr/bin/env python3
"""
Kernel Profile 收集工具

使用 PyTorch Profiler 收集 kernel 的性能指标，用于干扰感知调度。
分析每个 kernel 是 compute-bound 还是 memory-bound。
"""

import torch
import torch.profiler
from torch.profiler import ProfilerActivity
import json
import sys
import os

sys.path.insert(0, ".")
from GPT import create_char_gpt, sequence_len

def collect_kernel_profiles(model, input_tensor, num_warmup=3, num_profile=5):
    """
    收集模型推理过程中所有 kernel 的 profile 信息
    """
    print("=" * 70)
    print("COLLECTING KERNEL PROFILES")
    print("=" * 70)
    
    # Warmup
    print(f"Warmup ({num_warmup} iterations)...")
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(input_tensor)
    torch.cuda.synchronize()
    
    # Profile
    print(f"Profiling ({num_profile} iterations)...")
    
    with torch.profiler.profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        for _ in range(num_profile):
            with torch.no_grad():
                _ = model(input_tensor)
            torch.cuda.synchronize()
    
    return prof

def analyze_kernel_profiles(prof):
    """
    分析 kernel profiles，判断每个 kernel 是 compute-bound 还是 memory-bound
    """
    print("\n" + "=" * 70)
    print("ANALYZING KERNEL PROFILES")
    print("=" * 70)
    
    # 获取 kernel 统计
    kernel_stats = prof.key_averages()
    
    kernels = []
    
    for stat in kernel_stats:
        # 只分析 CUDA kernels
        if stat.device_type != torch.autograd.DeviceType.CUDA:
            continue
        
        if stat.count == 0:
            continue
        
        name = stat.key
        cuda_time_us = stat.cuda_time_total / stat.count  # 平均 CUDA 时间 (us)
        flops = stat.flops if hasattr(stat, 'flops') and stat.flops else 0
        
        # 计算 FLOPS/s (如果有 FLOPS 数据)
        flops_per_sec = 0
        if flops > 0 and cuda_time_us > 0:
            flops_per_sec = (flops / stat.count) / (cuda_time_us / 1e6)  # FLOPS/s
        
        # 判断类型
        # 简单启发式：如果有高 FLOPS，可能是 compute-bound
        # 更准确的方法需要使用 CUPTI 获取 SM utilization 和 memory throughput
        profile_type = "unknown"
        if flops_per_sec > 1e12:  # > 1 TFLOPS
            profile_type = "compute"
        elif "memcpy" in name.lower() or "memset" in name.lower():
            profile_type = "memory"
        elif "sgemm" in name.lower() or "gemm" in name.lower():
            profile_type = "compute"
        elif "softmax" in name.lower() or "elementwise" in name.lower():
            profile_type = "memory"
        
        kernel_info = {
            "name": name,
            "count": stat.count,
            "cuda_time_us": cuda_time_us,
            "cuda_time_total_us": stat.cuda_time_total,
            "flops": flops / stat.count if flops else 0,
            "flops_per_sec": flops_per_sec,
            "profile_type": profile_type,
            "self_cuda_memory_usage": stat.self_cuda_memory_usage if hasattr(stat, 'self_cuda_memory_usage') else 0,
        }
        kernels.append(kernel_info)
    
    # 按 CUDA 时间排序
    kernels.sort(key=lambda x: -x["cuda_time_total_us"])
    
    return kernels

def generate_profile_table(kernels, output_file="kernel_profiles.json"):
    """
    生成调度器使用的 profile 表
    """
    print("\n" + "=" * 70)
    print("GENERATING PROFILE TABLE")
    print("=" * 70)
    
    # 转换为调度器格式
    profile_entries = []
    
    for i, k in enumerate(kernels[:50]):  # 取 top 50 kernels
        # 估算 SM 需求（简化：基于执行时间）
        # 实际应使用 CUPTI 获取 achieved_occupancy
        sm_needed = 20  # 默认值
        if k["profile_type"] == "compute":
            sm_needed = 40  # compute-bound 通常需要更多 SM
        elif k["profile_type"] == "memory":
            sm_needed = 10  # memory-bound 不需要太多 SM
        
        entry = {
            "kernel_id": f"{k['name']}:{i}",
            "duration_ms": k["cuda_time_us"] / 1000.0,
            "sm_needed": sm_needed,
            "profile_type": k["profile_type"],
            "flops": k["flops"],
        }
        profile_entries.append(entry)
        
        print(f"  {k['name'][:50]:50s} | {k['cuda_time_us']:8.1f} us | {k['profile_type']:8s} | SM: {sm_needed}")
    
    # 保存
    output = {
        "version": "1.0",
        "model": "GPT",
        "kernels": profile_entries
    }
    
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to {output_file}")
    return output

def print_summary(kernels):
    """打印分析摘要"""
    print("\n" + "=" * 70)
    print("PROFILE SUMMARY")
    print("=" * 70)
    
    total_time = sum(k["cuda_time_total_us"] for k in kernels)
    
    compute_kernels = [k for k in kernels if k["profile_type"] == "compute"]
    memory_kernels = [k for k in kernels if k["profile_type"] == "memory"]
    unknown_kernels = [k for k in kernels if k["profile_type"] == "unknown"]
    
    compute_time = sum(k["cuda_time_total_us"] for k in compute_kernels)
    memory_time = sum(k["cuda_time_total_us"] for k in memory_kernels)
    unknown_time = sum(k["cuda_time_total_us"] for k in unknown_kernels)
    
    print(f"\nTotal unique kernels: {len(kernels)}")
    print(f"Total CUDA time: {total_time/1000:.2f} ms")
    print(f"\nBreakdown by type:")
    print(f"  Compute-bound: {len(compute_kernels):3d} kernels, {compute_time/1000:8.2f} ms ({100*compute_time/total_time:.1f}%)")
    print(f"  Memory-bound:  {len(memory_kernels):3d} kernels, {memory_time/1000:8.2f} ms ({100*memory_time/total_time:.1f}%)")
    print(f"  Unknown:       {len(unknown_kernels):3d} kernels, {unknown_time/1000:8.2f} ms ({100*unknown_time/total_time:.1f}%)")
    
    print("\n" + "=" * 70)
    print("TOP 10 COMPUTE-BOUND KERNELS")
    print("=" * 70)
    for k in compute_kernels[:10]:
        print(f"  {k['name'][:60]:60s} | {k['cuda_time_us']:8.1f} us")
    
    print("\n" + "=" * 70)
    print("TOP 10 MEMORY-BOUND KERNELS")
    print("=" * 70)
    for k in memory_kernels[:10]:
        print(f"  {k['name'][:60]:60s} | {k['cuda_time_us']:8.1f} us")

def main():
    print("Creating GPT model...")
    model = create_char_gpt(256, "cuda")
    model.eval()
    
    input_tensor = torch.randint(0, 256, (2, sequence_len), device="cuda")
    
    # 收集 profile
    prof = collect_kernel_profiles(model, input_tensor, num_warmup=3, num_profile=5)
    
    # 分析 kernels
    kernels = analyze_kernel_profiles(prof)
    
    # 打印摘要
    print_summary(kernels)
    
    # 生成 profile 表
    generate_profile_table(kernels, "profiles/kernel_profiles.json")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print("\nTo use with scheduler, load 'profiles/kernel_profiles.json'")
    print("and match kernel names to set sm_needed and profile_type.")

if __name__ == "__main__":
    main()
