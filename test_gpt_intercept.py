#!/usr/bin/env python3
"""
Test GPT model inference with real CUDA interception
"""
import os
import sys
import time
import ctypes

os.environ['ORION_LOG_LEVEL'] = '3'  # INFO

def main():
    print("=" * 60)
    print("GPT Model Inference with CUDA Interception")
    print("=" * 60)
    
    # Load PyTorch first
    import torch
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Import GPT model
    sys.path.insert(0, os.path.dirname(__file__))
    from GPT import create_char_gpt, sequence_len
    
    # Load scheduler library
    lib_path = os.path.join(os.path.dirname(__file__), "build", "libgpu_scheduler.so")
    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
    
    lib.orion_start_scheduler.argtypes = [ctypes.c_int]
    lib.orion_start_scheduler.restype = ctypes.c_int
    lib.orion_stop_scheduler.restype = None
    lib.orion_set_client_idx.argtypes = [ctypes.c_int]
    
    # Start scheduler
    print("\nStarting scheduler...")
    ret = lib.orion_start_scheduler(1)
    if ret != 0:
        print("Failed to start scheduler!")
        return 1
    
    try:
        # Create model
        device = "cuda"
        vocab_size = 256
        batch_size = 2
        
        print(f"\nCreating GPT model (batch_size={batch_size}, seq_len={sequence_len})...")
        model = create_char_gpt(vocab_size, device)
        model.eval()
        
        # Create input
        dummy_input = torch.randint(0, vocab_size, (batch_size, sequence_len), device=device)
        
        # Warmup
        print("Warming up...")
        with torch.no_grad():
            for _ in range(2):
                _ = model(dummy_input)
        torch.cuda.synchronize()
        
        # Set client_idx for scheduled inference
        print("Setting client_idx for scheduled inference...")
        lib.orion_set_client_idx(0)
        
        # Inference test
        num_iters = 5
        print(f"\nRunning {num_iters} inference iterations...")
        
        start_time = time.perf_counter()
        with torch.no_grad():
            for i in range(num_iters):
                output = model(dummy_input)
                torch.cuda.synchronize()
        end_time = time.perf_counter()
        
        elapsed = end_time - start_time
        throughput = num_iters / elapsed
        latency = elapsed / num_iters * 1000
        
        print(f"\n{'='*40}")
        print(f"Results:")
        print(f"  Total time: {elapsed:.3f} s")
        print(f"  Throughput: {throughput:.2f} iter/s")  
        print(f"  Latency: {latency:.2f} ms/iter")
        print(f"  Output shape: {output.shape}")
        print(f"{'='*40}")
        
        success = True
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Stop scheduler
    print("\nStopping scheduler...")
    lib.orion_stop_scheduler()
    
    print("\n" + "=" * 60)
    print("TEST PASSED" if success else "TEST FAILED")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
