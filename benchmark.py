import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader

# IMPORT YOUR MODULES
from src.models.lit_vit import LitQuantizedViT
from src.data import CIFAR100DataModule

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Path to the checkpoint you just trained (Check lightning_logs/version_X/checkpoints/)
CHECKPOINT_PATH = "/teamspace/studios/this_studio/Q-ViT_by_methodist/checkpoints/qvit-cifar100-run1/epoch_epoch=01_val_acc_val_acc=0.8936.ckpt" 
# ^^^ UPDATE THIS PATH ^^^

ONNX_PATH = "qvit_int8.onnx"
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Starting Benchmark on {DEVICE}")

# ==========================================
# 2. DATA PREPARATION
# ==========================================
print("--- Preparing Data ---")
dm = CIFAR100DataModule(batch_size=BATCH_SIZE)
dm.prepare_data()
dm.setup()
test_loader = dm.test_dataloader()

# ==========================================
# 3. LOAD MODELS
# ==========================================
print(f"--- Loading Checkpoint: {CHECKPOINT_PATH} ---")

# Load the entire Lightning Module
# strict=False allows loading even if some minor keys (like loss history) are missing
lit_model = LitQuantizedViT.load_from_checkpoint(CHECKPOINT_PATH, strict=False)
lit_model.eval()

# A. EXTRACT TEACHER (FP32 Baseline)
teacher_model = lit_model.teacher
teacher_model.to(DEVICE)
teacher_model.eval()
print("✅ Teacher Model Extracted (FP32)")

# B. EXTRACT STUDENT (Quantized)
student_model = lit_model.model
student_model.to("cpu") # Move to CPU for export
student_model.eval()
print("✅ Student Model Extracted")

# ==========================================
# 4. REAL QUANTIZATION: EXPORT TO ONNX
# ==========================================
print("\n--- Exporting Student to ONNX (Real INT8 Representation) ---")
dummy_input = torch.randn(1, 3, 224, 224)

# Dynamic axes allow variable batch sizes
dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}

# CONTEXT MANAGER: This disables the 'smart' fused kernel that requires Opset 14+
# It forces PyTorch to use standard MatMul/Softmax, which Opset 13 supports.
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    torch.onnx.export(
        student_model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=13,  # <--- You can keep this at 13 now!
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )
    
# Measure Real File Sizes
fp32_size = os.path.getsize(CHECKPOINT_PATH) / 1e6  # Approx size of full checkpoint
# For a fairer comparison, save just the teacher state dict
torch.save(teacher_model.state_dict(), "temp_teacher.pth")
real_fp32_size = os.path.getsize("temp_teacher.pth") / 1e6
os.remove("temp_teacher.pth")

int8_size = os.path.getsize(ONNX_PATH) / 1e6

print(f"📦 Teacher (FP32) Size: {real_fp32_size:.2f} MB")
print(f"📦 Student (ONNX) Size: {int8_size:.2f} MB")
print(f"📉 Size Reduction: {real_fp32_size / int8_size:.2f}x")

# ==========================================
# 5. INFERENCE ENGINES
# ==========================================

def run_pytorch_inference(model, loader):
    preds, targets, latencies = [], [], []
    model.to(DEVICE)
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data = data.to(DEVICE)
            
            # Synchronize for accurate GPU timing
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            start = time.time()
            
            output = model(data).logits # HuggingFace models return .logits
            
            if DEVICE.type == 'cuda': torch.cuda.synchronize()
            end = time.time()
            
            latencies.append((end - start) * 1000) # ms
            preds.extend(output.argmax(dim=1).cpu().numpy())
            targets.extend(target.numpy())
            
            if batch_idx % 50 == 0: print(f"Teacher Batch {batch_idx}...")
            
    return preds, targets, np.mean(latencies)

def run_onnx_inference(onnx_file, loader):
    preds, targets, latencies = [], [], []
    
    # Use CUDA Provider if available, else CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_file, providers=providers)
    input_name = session.get_inputs()[0].name
    
    print(f"ONNX Runtime Providers: {session.get_providers()}")
    
    for batch_idx, (data, target) in enumerate(loader):
        # ONNX Runtime requires numpy inputs
        ort_inputs = {input_name: data.numpy()}
        
        start = time.time()
        ort_outs = session.run(None, ort_inputs)
        end = time.time()
        
        latencies.append((end - start) * 1000) # ms
        
        # Output is usually a list, take the first element
        pred_batch = np.argmax(ort_outs[0], axis=1)
        preds.extend(pred_batch)
        targets.extend(target.numpy())
        
        if batch_idx % 50 == 0: print(f"Student Batch {batch_idx}...")
        
    return preds, targets, np.mean(latencies)

# ==========================================
# 6. RUN BENCHMARKS
# ==========================================

# A. Teacher Benchmark
t_preds, t_targets, t_lat = run_pytorch_inference(teacher_model, test_loader)

# B. Student Benchmark
s_preds, s_targets, s_lat = run_onnx_inference(ONNX_PATH, test_loader)

# ==========================================
# 7. REPORTING & VISUALIZATION
# ==========================================
def compute_metrics(preds, targets):
    return {
        "acc": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average='macro'),
        "f1_weighted": f1_score(targets, preds, average='weighted'),
        "cm": confusion_matrix(targets, preds)
    }

t_metrics = compute_metrics(t_preds, t_targets)
s_metrics = compute_metrics(s_preds, s_targets)

print("\n" + "="*60)
print("FINAL BENCHMARK REPORT (CIFAR-100)")
print("="*60)
print(f"{'Metric':<25} | {'Teacher (FP32)':<15} | {'Student (INT8)':<15} | {'Delta'}")
print("-" * 75)
print(f"{'Size (MB)':<25} | {real_fp32_size:<15.2f} | {int8_size:<15.2f} | {real_fp32_size/int8_size:.2f}x Smaller")
print(f"{'Latency (ms/batch)':<25} | {t_lat:<15.2f} | {s_lat:<15.2f} | {t_lat/s_lat:.2f}x Faster")
print(f"{'Accuracy':<25} | {t_metrics['acc']:<15.4f} | {s_metrics['acc']:<15.4f} | {s_metrics['acc']-t_metrics['acc']:.4f}")
print(f"{'F1 (Macro)':<25} | {t_metrics['f1_macro']:<15.4f} | {s_metrics['f1_macro']:<15.4f} | {s_metrics['f1_macro']-t_metrics['f1_macro']:.4f}")
print("-" * 75)

# Plot Confusion Matrices
fig, ax = plt.subplots(1, 2, figsize=(24, 10))
sns.heatmap(t_metrics['cm'], ax=ax[0], cmap='Blues', annot=False)
ax[0].set_title("Teacher Confusion Matrix")
ax[0].set_xlabel("Predicted")
ax[0].set_ylabel("True")

sns.heatmap(s_metrics['cm'], ax=ax[1], cmap='Greens', annot=False)
ax[1].set_title(f"Student (Quantized) Confusion Matrix\nAccuracy: {s_metrics['acc']:.4f}")
ax[1].set_xlabel("Predicted")
ax[1].set_ylabel("True")

plt.tight_layout()
plt.savefig("benchmark_comparison.png")
print("\n✅ Results saved to 'benchmark_comparison.png'")