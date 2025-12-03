#!/usr/bin/env python3
"""
多客户端 PyTorch Profiler 分析 V2
修复：确保记录 CPU 操作
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
    
    # 用于同步的事件
    start_event = threading.Event()
    done_events = [threading.Event() for _ in range(NUM_CLIENTS)]
    results = [None] * NUM_CLIENTS
    
    def client_worker(client_idx, num_iters):
        """客户端工作线程"""
        # 等待开始信号
        start_event.wait()
        
        # 设置客户端索引
        lib.orion_set_client_idx(client_idx)
        client_type = "HP" if client_idx == 0 else f"BE{client_idx}"
        
        for i in range(num_iters):
            with torch.no_grad():
                output = model(inputs[client_idx])
            torch.cuda.synchronize()
        
        results[client_idx] = output.shape
        done_events[client_idx].set()
    
    # Warmup (passthrough)
    print("Warmup (passthrough)...")
    for _ in range(2):
        with torch.no_grad():
            _ = model(inputs[0])
    torch.cuda.synchronize()
    
    # 创建线程但不启动
    threads = []
    for i in range(NUM_CLIENTS):
        num_iters = 3 if i == 0 else 2
        t = threading.Thread(target=client_worker, args=(i, num_iters))
        threads.append(t)
    
    # 先启动所有线程（它们会等待 start_event）
    for t in threads:
        t.start()
    
    time.sleep(0.1)  # 确保所有线程都在等待
    
    # Profile - 在主线程启动 profiler，然后触发其他线程
    print(f"Profiling with {NUM_CLIENTS} concurrent clients...")
    
    with torch.profiler.profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        experimental_config=torch.profiler._ExperimentalConfig(
            profiler_metrics=[],
            profiler_measure_per_kernel=False,
        ),
    ) as prof:
        # 触发所有客户端线程开始工作
        start_event.set()
        
        # 同时在主线程也做一些工作以确保记录 CPU ops
        lib.orion_set_client_idx(0)  # 主线程作为 HP client
        print("  Main thread (HP) running...")
        for i in range(2):
            with torch.no_grad():
                with torch.profiler.record_function("HP_forward_pass"):
                    output = model(inputs[0])
            torch.cuda.synchronize()
        
        # 等待所有客户端完成
        for i, event in enumerate(done_events):
            event.wait()
            client_type = "HP" if i == 0 else f"BE{i}"
            print(f"  {client_type} completed")
    
    # 等待线程结束
    for t in threads:
        t.join()
    
    # 导出
    prof.export_chrome_trace("profiles/multiclient_v2.json")
    
    # 打印结果
    print("\n--- Results ---")
    for i in range(NUM_CLIENTS):
        client_type = "HP" if i == 0 else f"BE{i}"
        print(f"  {client_type}: output shape = {results[i]}")
    
    # 打印统计
    print("\n--- Top 15 Operations ---")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
    
    # 保存统计
    with open("profiles/multiclient_v2_stats.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"MULTI-CLIENT EXECUTION PROFILE ({NUM_CLIENTS} clients)\n")
        f.write("=" * 80 + "\n\n")
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    
    lib.orion_stop_scheduler()
    print("\nSaved: profiles/multiclient_v2.json, profiles/multiclient_v2_stats.txt")
    
    return True

def analyze_and_fix():
    """分析并修复线程 ID"""
    print("\n" + "=" * 70)
    print("ANALYZING AND FIXING THREAD IDs")
    print("=" * 70)
    
    with open("profiles/multiclient_v2.json") as f:
        data = json.load(f)
    
    # 统计类别
    categories = {}
    for e in data["traceEvents"]:
        cat = e.get("cat", "no_cat")
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nEvent categories:")
    for cat, cnt in sorted(categories.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {cnt}")
    
    # 统计线程的 CUDA 活动
    threads = {}
    for e in data["traceEvents"]:
        if e.get("cat") == "cuda_runtime":
            tid = e.get("tid")
            name = e.get("name")
            if tid not in threads:
                threads[tid] = {"total": 0, "launch": 0}
            threads[tid]["total"] += 1
            if name == "cudaLaunchKernel":
                threads[tid]["launch"] += 1
    
    print("\nCUDA activity by thread:")
    for tid, stats in sorted(threads.items(), key=lambda x: -x[1]["launch"]):
        print(f"  tid={tid}: {stats['launch']} launches, {stats['total']} total")
    
    # 找到调度器线程（最多 launch 的）
    scheduler_tid = max(threads.keys(), key=lambda t: threads[t]["launch"])
    
    # 转换负数线程 ID
    tid_mapping = {}
    next_id = 100000
    for tid in threads.keys():
        if tid < 0:
            tid_mapping[tid] = next_id
            next_id += 1
    
    for event in data["traceEvents"]:
        old_tid = event.get("tid")
        if old_tid in tid_mapping:
            event["tid"] = tid_mapping[old_tid]
    
    # 找到主进程 ID
    main_pid = None
    for e in data["traceEvents"]:
        if e.get("name") == "process_name":
            main_pid = e.get("pid")
            break
    if main_pid is None:
        main_pid = list(threads.keys())[0]
    
    # 添加线程标签
    metadata = [
        {"name": "thread_name", "ph": "M", "pid": main_pid, "tid": scheduler_tid,
         "args": {"name": ">>> SCHEDULER THREAD <<<"}},
        {"name": "thread_sort_index", "ph": "M", "pid": main_pid, "tid": scheduler_tid,
         "args": {"sort_index": 0}},
    ]
    
    # 客户端线程
    for i, (old_tid, new_tid) in enumerate(sorted(tid_mapping.items())):
        metadata.append({
            "name": "thread_name", "ph": "M", "pid": main_pid, "tid": new_tid,
            "args": {"name": f">>> CLIENT {i} <<<"}
        })
    
    data["traceEvents"].extend(metadata)
    
    with open("profiles/multiclient_v2_final.json", "w") as f:
        json.dump(data, f)
    
    print(f"\nScheduler thread: {scheduler_tid}")
    print(f"Client threads: {list(tid_mapping.values())}")
    print(f"Saved: profiles/multiclient_v2_final.json")

def main():
    os.makedirs("profiles", exist_ok=True)
    
    if profile_multiclient():
        analyze_and_fix()
    
    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()
