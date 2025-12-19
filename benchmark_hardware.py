"""
Hardware Benchmarking Script for Q-ViT
Measures Params, FLOPs, and Inference Throughput on available hardware.
"""
import torch
import time
import psutil
import os
from src.models import LitQuantizedViT
from thop import profile, clever_format  # Requires: pip install thop

def benchmark():
    print("="*60)
    print("Q-ViT HARDWARE BENCHMARK")
    print("="*60)
    
    # 1. Hardware Detection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        caps = torch.cuda.get_device_capability(0)
        print(f"GPU Model: {gpu_name}")
        print(f"VRAM: {vram_gb:.2f} GB")
        print(f"Compute Capability: {caps}")
    
    # 2. Model Loading
    print("\n[Initializing Model...]")
    # Initialize with 4-bit config
    model = LitQuantizedViT(
        model_name="google/vit-large-patch16-224",
        nbits_w=4,
        nbits_a=4,
        num_classes=100
    ).to(device)
    model.eval()
    
    # 3. Complexity Metrics (Params & FLOPs)
    print("\n[Measuring Complexity...]")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Using thop to calculate FLOPs
    # Note: These are theoretical FLOPs for the architecture. 
    # Since we use fake quantization, actual execution is FP16/FP32.
    macs, params = profile(model.model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2 # 1 MAC = 2 FLOPs
    
    flops_str, params_str = clever_format([flops, params], "%.3f")
    print(f"Parameters: {params_str}")
    print(f"FLOPs (per image): {flops_str}")
    
    # 4. Throughput Benchmarking
    print("\n[Benchmarking Inference Speed...]")
    batch_sizes = [1, 16, 32, 64, 128]
    
    print(f"{'Batch Size':<12} | {'Latency (ms)':<15} | {'Throughput (img/s)':<20} | {'Memory (MB)':<15}")
    print("-" * 75)
    
    for bs in batch_sizes:
        try:
            input_batch = torch.randn(bs, 3, 224, 224).to(device)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(input_batch)
            
            torch.cuda.synchronize()
            start_time = time.time()
            iterations = 50
            
            with torch.no_grad():
                for _ in range(iterations):
                    _ = model(input_batch)
                    
            torch.cuda.synchronize()
            end_time = time.time()
            
            total_time = end_time - start_time
            avg_batch_time = total_time / iterations
            throughput = (bs * iterations) / total_time
            
            # Memory Usage
            mem_used = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == 'cuda' else 0
            torch.cuda.reset_peak_memory_stats()
            
            print(f"{bs:<12} | {avg_batch_time*1000:<15.2f} | {throughput:<20.2f} | {mem_used:<15.0f}")
            
        except RuntimeError as e:
            print(f"{bs:<12} | OOM or Error: {str(e)[:30]}...")

    print("\n" + "="*60)
    print("NOTE: This implementation uses 'Fake Quantization'.")
    print("Operations are simulated in FP16/FP32. Real speedup on A100 requires")
    print("writing custom CUDA INT4 kernels or using TensorRT.")
    print("="*60)

if __name__ == "__main__":
    benchmark()