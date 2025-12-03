#!/usr/bin/env python3
"""
多客户端 PyTorch Profiler 分析
测试不同客户端线程使用不同 CUDA stream
"""

import torch
import torch.profiler
from torch.profiler import ProfilerActivity
import ctypes
import os
import sys
import json
import threading
import time

sys.path.insert(0, ".")
from GPT import create_char_gpt, sequence_len

NUM_CLIENTS = 3  # 1 HP + 2 BE

def profile_multiclient():
    """多客户端调度执行的 profile"""
    print("=" * 70)
    print(f"PROFILING MULTI-CLIENT EXECUTION ({NUM_CLIENTS} clients)")
    print("=" * 70)
    
    # 加载调度器
    lib = ctypes.CDLL("./build/libgpu_scheduler.so", mode=ctypes.RTLD_LOCAL)
    lib.orion_start_scheduler(NUM_CLIENTS)
    
    # 在 passthrough 模式下创建模型
    print("Creating GPT model (passthrough mode)...")
    model = create_char_gpt(256, "cuda")
    model.eval()
    
    # 每个客户端的输入
    inputs = [torch.randint(0, 256, (2, sequence_len), device="cuda") for _ in range(NUM_CLIENTS)]
    torch.cuda.synchronize()
    
    # 客户端线程结果
    results = [None] * NUM_CLIENTS
    errors = [None] * NUM_CLIENTS
    
    def client_worker(client_idx, num_iters):
        """客户端工作线程"""
        try:
            # 设置客户端索引
            lib.orion_set_client_idx(client_idx)
            client_type = "HP" if client_idx == 0 else f"BE{client_idx}"
            print(f"  {client_type} (client {client_idx}): starting {num_iters} iterations")
            
            for i in range(num_iters):
                with torch.no_grad():
                    output = model(inputs[client_idx])
                torch.cuda.synchronize()
            
            results[client_idx] = output.shape
            print(f"  {client_type} (client {client_idx}): completed")
        except Exception as e:
            errors[client_idx] = str(e)
            print(f"  Client {client_idx} error: {e}")
    
    # Warmup (passthrough)
    print("Warmup (passthrough)...")
    for _ in range(2):
        with torch.no_grad():
            _ = model(inputs[0])
    torch.cuda.synchronize()
    
    # Profile with multiple clients
    print(f"Profiling with {NUM_CLIENTS} concurrent clients...")
    
    with torch.profiler.profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # 启动所有客户端线程
        threads = []
        for i in range(NUM_CLIENTS):
            # HP 做更多迭代
            num_iters = 3 if i == 0 else 2
            t = threading.Thread(target=client_worker, args=(i, num_iters))
            threads.append(t)
        
        # 启动线程（稍微错开）
        for i, t in enumerate(threads):
            t.start()
            time.sleep(0.05)  # 50ms 间隔
        
        # 等待所有线程完成
        for t in threads:
            t.join()
    
    # 导出
    prof.export_chrome_trace("profiles/multiclient_scheduled.json")
    
    # 打印结果
    print("\n--- Results ---")
    for i in range(NUM_CLIENTS):
        client_type = "HP" if i == 0 else f"BE{i}"
        if errors[i]:
            print(f"  {client_type}: ERROR - {errors[i]}")
        else:
            print(f"  {client_type}: output shape = {results[i]}")
    
    # 保存统计
    with open("profiles/multiclient_stats.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"MULTI-CLIENT EXECUTION PROFILE ({NUM_CLIENTS} clients)\n")
        f.write("=" * 80 + "\n\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
    
    lib.orion_stop_scheduler()
    print("\nSaved: profiles/multiclient_scheduled.json, profiles/multiclient_stats.txt")
    
    return True

def analyze_threads():
    """分析线程分布"""
    print("\n" + "=" * 70)
    print("THREAD ANALYSIS")
    print("=" * 70)
    
    filepath = "profiles/multiclient_scheduled.json"
    if not os.path.exists(filepath):
        print("Profile file not found!")
        return
    
    with open(filepath) as f:
        data = json.load(f)
    
    # 统计线程
    threads = {}
    for event in data.get("traceEvents", []):
        tid = event.get("tid")
        cat = event.get("cat", "")
        name = event.get("name", "")
        if tid is not None and cat == "cuda_runtime":
            if tid not in threads:
                threads[tid] = {"total": 0, "calls": {}}
            threads[tid]["total"] += 1
            threads[tid]["calls"][name] = threads[tid]["calls"].get(name, 0) + 1
    
    print(f"\nFound {len(threads)} threads with CUDA activity:")
    for tid, stats in sorted(threads.items(), key=lambda x: -x[1]["total"]):
        launch = stats["calls"].get("cudaLaunchKernel", 0)
        print(f"\n  Thread {tid}: {stats['total']} CUDA calls ({launch} cudaLaunchKernel)")
        for name, cnt in sorted(stats["calls"].items(), key=lambda x: -x[1])[:5]:
            print(f"    {name}: {cnt}")
    
    # 转换线程 ID 为正数并添加标签
    print("\n" + "=" * 70)
    print("CREATING LABELED VERSION")
    print("=" * 70)
    
    # 找到调度器线程（负数 tid，最多 cudaLaunchKernel）
    scheduler_tid = None
    max_launches = 0
    for tid, stats in threads.items():
        if tid < 0:
            launches = stats["calls"].get("cudaLaunchKernel", 0)
            if launches > max_launches:
                max_launches = launches
                scheduler_tid = tid
    
    if scheduler_tid:
        NEW_SCHEDULER_TID = 99999
        count = 0
        for event in data["traceEvents"]:
            if event.get("tid") == scheduler_tid:
                event["tid"] = NEW_SCHEDULER_TID
                count += 1
        print(f"  Converted scheduler thread {scheduler_tid} -> {NEW_SCHEDULER_TID} ({count} events)")
        
        # 添加线程元数据
        client_tids = [tid for tid in threads.keys() if tid > 0]
        pid = client_tids[0] if client_tids else 0
        
        metadata = [
            {'name': 'thread_name', 'ph': 'M', 'pid': pid, 'tid': NEW_SCHEDULER_TID,
             'args': {'name': '>>> SCHEDULER THREAD <<<'}},
            {'name': 'thread_sort_index', 'ph': 'M', 'pid': pid, 'tid': NEW_SCHEDULER_TID,
             'args': {'sort_index': 0}},
        ]
        
        for i, tid in enumerate(sorted(client_tids)):
            metadata.append({
                'name': 'thread_name', 'ph': 'M', 'pid': pid, 'tid': tid,
                'args': {'name': f'>>> CLIENT THREAD {i} <<<'}
            })
            metadata.append({
                'name': 'thread_sort_index', 'ph': 'M', 'pid': pid, 'tid': tid,
                'args': {'sort_index': i + 1}
            })
        
        data["traceEvents"].extend(metadata)
        
        with open("profiles/multiclient_final.json", "w") as f:
            json.dump(data, f)
        print(f"  Saved: profiles/multiclient_final.json")

def main():
    os.makedirs("profiles", exist_ok=True)
    
    if profile_multiclient():
        analyze_threads()
    
    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)
    print("\nTo view in Chrome:")
    print("  1. Open chrome://tracing")
    print("  2. Load profiles/multiclient_final.json")

if __name__ == "__main__":
    main()
