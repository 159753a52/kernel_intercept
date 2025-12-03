#!/usr/bin/env python3
"""
Test script for Orion-style GPU scheduler with GPT model

Usage:
    python test_gpt.py           # Test with scheduler (no interception)
    python test_gpt.py --baseline # Test without scheduler
    python test_gpt.py --both    # Run both tests
"""

import os
import sys
import ctypes
import time

# 设置环境变量
os.environ['ORION_LOG_LEVEL'] = '3'  # INFO

def load_scheduler_lib():
    """加载调度器库"""
    lib_path = os.path.join(os.path.dirname(__file__), "build", "libgpu_scheduler.so")
    
    if not os.path.exists(lib_path):
        print(f"Error: Library not found at {lib_path}")
        print("Please run 'make' first to build the library")
        return None
    
    try:
        # 使用 RTLD_LOCAL 避免符号冲突 (不导出符号给其他库)
        lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
        
        # 设置函数签名
        lib.orion_init.argtypes = [ctypes.c_int]
        lib.orion_init.restype = ctypes.c_int
        
        lib.orion_shutdown.argtypes = []
        lib.orion_shutdown.restype = None
        
        lib.orion_set_client_idx.argtypes = [ctypes.c_int]
        lib.orion_set_client_idx.restype = None
        
        lib.orion_start_scheduler.argtypes = [ctypes.c_int]
        lib.orion_start_scheduler.restype = ctypes.c_int
        
        lib.orion_stop_scheduler.argtypes = []
        lib.orion_stop_scheduler.restype = None
        
        lib.orion_set_enabled.argtypes = [ctypes.c_int]
        lib.orion_set_enabled.restype = None
        
        lib.block.argtypes = [ctypes.c_int]
        lib.block.restype = None
        
        print(f"[OK] Loaded scheduler library from {lib_path}")
        return lib
        
    except Exception as e:
        print(f"Error loading library: {e}")
        return None


def test_gpt_inference():
    """测试 GPT 模型推理"""
    print("\n" + "="*60)
    print("Testing GPT Model Inference with Orion Scheduler")
    print("="*60)
    
    # 加载调度器库
    lib = load_scheduler_lib()
    if not lib:
        return False
    
    # 导入 PyTorch 和 GPT 模型
    try:
        import torch
        print(f"[OK] PyTorch version: {torch.__version__}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("[ERROR] PyTorch not installed")
        return False
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        return False
    
    # 从 GPT.py 导入模型
    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from GPT import create_char_gpt, sequence_len
        print("[OK] Imported GPT model")
    except ImportError as e:
        print(f"[ERROR] Failed to import GPT model: {e}")
        return False
    
    # 初始化调度器 (1 个 HP client)
    print("\n[INFO] Initializing scheduler with 1 client...")
    ret = lib.orion_init(1)
    if ret != 0:
        print("[ERROR] Failed to init capture layer")
        return False
    
    # 禁用拦截 - 只测试调度器基础设施
    lib.orion_set_enabled(0)
    print("[OK] Capture layer initialized (interception disabled)")
    
    # 设置当前线程为 client 0 (HP)
    lib.orion_set_client_idx(0)
    print("[OK] Set current thread as HP client (idx=0)")
    
    try:
        device = "cuda:0"
        vocab_size = 256
        batch_size = 2
        
        print(f"\n[INFO] Creating GPT model...")
        print(f"  - vocab_size: {vocab_size}")
        print(f"  - sequence_len: {sequence_len}")
        print(f"  - batch_size: {batch_size}")
        
        # 创建模型
        model = create_char_gpt(vocab_size, device)
        model.eval()
        print("[OK] Model created and moved to GPU")
        
        # 创建输入数据
        dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_len), device=device)
        print(f"[OK] Created dummy input: shape={dummy_input.shape}")
        
        # 预热
        print("\n[INFO] Warming up...")
        with torch.no_grad():
            for i in range(3):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        print("[OK] Warmup complete")
        
        # 性能测试
        num_iters = 10
        print(f"\n[INFO] Running {num_iters} inference iterations...")
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            for i in range(num_iters):
                output = model(dummy_input)
        
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        throughput = num_iters / elapsed
        latency = elapsed / num_iters * 1000
        
        print(f"\n[RESULTS]")
        print(f"  - Total time: {elapsed:.3f} s")
        print(f"  - Throughput: {throughput:.2f} iter/s")
        print(f"  - Avg latency: {latency:.2f} ms")
        print(f"  - Output shape: {output.shape}")
        
        success = True
        
    except Exception as e:
        print(f"\n[ERROR] Exception during inference: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    finally:
        # 关闭调度器
        print("\n[INFO] Shutting down...")
        lib.orion_shutdown()
        print("[OK] Shutdown complete")
    
    return success


def test_without_scheduler():
    """不使用调度器的基线测试"""
    print("\n" + "="*60)
    print("Baseline Test (No Scheduler)")
    print("="*60)
    
    try:
        import torch
        sys.path.insert(0, os.path.dirname(__file__))
        from GPT import create_char_gpt, sequence_len
    except ImportError as e:
        print(f"[ERROR] Import failed: {e}")
        return False
    
    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available")
        return False
    
    device = "cuda:0"
    vocab_size = 256
    batch_size = 2
    
    model = create_char_gpt(vocab_size, device)
    model.eval()
    
    dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_len), device=device)
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # 测试
    num_iters = 10
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(num_iters):
            output = model(dummy_input)
    
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    throughput = num_iters / elapsed
    latency = elapsed / num_iters * 1000
    
    print(f"\n[BASELINE RESULTS]")
    print(f"  - Total time: {elapsed:.3f} s")
    print(f"  - Throughput: {throughput:.2f} iter/s")
    print(f"  - Avg latency: {latency:.2f} ms")
    
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', action='store_true', help='Run baseline test only')
    parser.add_argument('--both', action='store_true', help='Run both baseline and scheduler test')
    args = parser.parse_args()
    
    if args.baseline:
        success = test_without_scheduler()
    elif args.both:
        success = test_without_scheduler() and test_gpt_inference()
    else:
        success = test_gpt_inference()
    
    print("\n" + "="*60)
    if success:
        print("TEST PASSED")
    else:
        print("TEST FAILED")
    print("="*60)
    
    sys.exit(0 if success else 1)
