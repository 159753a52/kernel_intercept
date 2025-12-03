#!/usr/bin/env python3
"""
Minimal test to debug CUDA interception
"""
import os
import sys
import ctypes
import time

os.environ['ORION_LOG_LEVEL'] = '4'  # DEBUG

def main():
    print("=" * 60)
    print("CUDA Interception Debug Test")
    print("=" * 60)
    
    # Step 1: Load PyTorch and initialize CUDA first
    print("\n[Step 1] Loading PyTorch and initializing CUDA...")
    import torch
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return 1
    
    # Force CUDA initialization
    _ = torch.zeros(1, device='cuda')
    torch.cuda.synchronize()
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Device: {torch.cuda.get_device_name(0)}")
    
    # Step 2: Load scheduler library AFTER PyTorch
    print("\n[Step 2] Loading scheduler library...")
    lib_path = os.path.join(os.path.dirname(__file__), "build", "libgpu_scheduler.so")
    if not os.path.exists(lib_path):
        print(f"Library not found: {lib_path}")
        return 1
    
    lib = ctypes.CDLL(lib_path, mode=ctypes.RTLD_LOCAL)
    
    # Setup function signatures
    lib.orion_init.argtypes = [ctypes.c_int]
    lib.orion_init.restype = ctypes.c_int
    lib.orion_shutdown.restype = None
    lib.orion_set_client_idx.argtypes = [ctypes.c_int]
    lib.orion_set_enabled.argtypes = [ctypes.c_int]
    lib.orion_start_scheduler.argtypes = [ctypes.c_int]
    lib.orion_start_scheduler.restype = ctypes.c_int
    lib.orion_stop_scheduler.restype = None
    
    print("  Library loaded successfully")
    
    # Step 3: Initialize and start scheduler
    print("\n[Step 3] Starting scheduler with 1 client...")
    ret = lib.orion_start_scheduler(1)
    if ret != 0:
        print("  Failed to start scheduler!")
        return 1
    print("  Scheduler started")
    
    # Step 4: Set this thread as client 0
    print("\n[Step 4] Setting current thread as client 0...")
    lib.orion_set_client_idx(0)
    print("  Done")
    
    # Step 5: Run a simple CUDA operation
    print("\n[Step 5] Running simple CUDA operations...")
    
    try:
        # Simple tensor operations
        a = torch.randn(100, 100, device='cuda')
        b = torch.randn(100, 100, device='cuda')
        
        print("  Created tensors on GPU")
        
        # Matrix multiply (uses cuBLAS)
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        print(f"  Matrix multiply done: {c.shape}")
        
        # More operations
        for i in range(5):
            c = torch.mm(a, b)
            torch.cuda.synchronize()
        print(f"  5 more matrix multiplies done")
        
        success = True
        
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # Step 6: Cleanup
    print("\n[Step 6] Stopping scheduler...")
    lib.orion_stop_scheduler()
    print("  Done")
    
    print("\n" + "=" * 60)
    print("TEST PASSED" if success else "TEST FAILED")
    print("=" * 60)
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
