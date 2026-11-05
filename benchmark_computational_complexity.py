"""
benchmark_computational_complexity.py
Rigorous profiling for:
- Classic U-Net
- UNet++
- RepSegNet V1
- RepSegNet V2 (Proposed)

Measures:
1. Trainable Parameters (M)
2. Computational Complexity (GFLOPs) at 512x512
3. GPU Inference Latency (ms per image) & Throughput (FPS)
4. CPU Inference Latency (ms per image)
5. Training time per epoch (seconds)
6. Total training time & complexity with vs. without hyperparameter tuning
"""

import gc
import time
import json
import torch
import numpy as np
from thop import profile

from config import DEVICE, SEG_ENCODER_NAME, SEG_IMG_SIZE
from unet import get_unet
from unet_plus_plus import get_unetpp
from repsegnet import get_repsegnet
from repsegnet_v2 import get_repsegnet_v2

def benchmark_single_model(name, model_fn):
    print(f"\n==========================================", flush=True)
    print(f"Benchmarking: {name}", flush=True)
    print(f"==========================================", flush=True)

    model = model_fn().to(DEVICE)
    model.eval()

    # 1. Parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # 2. GFLOPs
    dummy_input_cuda = torch.randn(1, 3, SEG_IMG_SIZE, SEG_IMG_SIZE, device=DEVICE)
    flops, _ = profile(model, inputs=(dummy_input_cuda,), verbose=False)
    gflops = flops / 1e9

    # 3. GPU Latency
    for _ in range(15):
        with torch.no_grad():
            _ = model(dummy_input_cuda)
    torch.cuda.synchronize()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 100
    timings = np.zeros(repetitions)

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input_cuda)
            ender.record()
            torch.cuda.synchronize()
            timings[rep] = starter.elapsed_time(ender)

    gpu_latency_ms = float(np.mean(timings))
    gpu_std_ms = float(np.std(timings))
    gpu_fps = 1000.0 / gpu_latency_ms

    # 4. CPU Latency
    model.to('cpu')
    model.eval()
    dummy_input_cpu = torch.randn(1, 3, SEG_IMG_SIZE, SEG_IMG_SIZE, device='cpu')
    for _ in range(2):
        with torch.no_grad():
            _ = model(dummy_input_cpu)

    cpu_timings = []
    with torch.no_grad():
        for _ in range(5):
            t0 = time.perf_counter()
            _ = model(dummy_input_cpu)
            cpu_timings.append((time.perf_counter() - t0) * 1000.0)
    cpu_latency_ms = float(np.mean(cpu_timings))

    # 5. Training Step Timing
    model.to(DEVICE)
    model.train()
    b_size = 4
    train_batch = torch.randn(b_size, 3, SEG_IMG_SIZE, SEG_IMG_SIZE, device=DEVICE)
    target_batch = torch.randn(b_size, 1, SEG_IMG_SIZE, SEG_IMG_SIZE, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    for _ in range(3):
        optimizer.zero_grad()
        out = model(train_batch)
        if isinstance(out, (tuple, list)):
            out = out[0]
        loss = criterion(out, target_batch)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    step_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        out = model(train_batch)
        if isinstance(out, (tuple, list)):
            out = out[0]
        loss = criterion(out, target_batch)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)

    # In BUSI, 546 training images with batch_size=4 is ~136 steps/epoch
    sec_per_step = float(np.mean(step_times))
    sec_per_epoch = sec_per_step * 136.0

    # Without Hyperparameter Tuning: 50 epochs standard single training run
    train_time_no_tune_min = (sec_per_epoch * 50.0) / 60.0

    # With Hyperparameter Tuning:
    # Bayesian Optimization / Grid Search across learning rates, loss weights (CompoundLoss),
    # and attention variants: typical standard protocol is 15 exploration trials x 50 epochs
    num_tuning_trials = 15
    train_time_with_tune_min = (sec_per_epoch * 50.0 * num_tuning_trials) / 60.0
    train_time_with_tune_hours = train_time_with_tune_min / 60.0

    print(f"  Trainable Params:    {trainable_params/1e6:.2f} M", flush=True)
    print(f"  Complexity:          {gflops:.2f} GFLOPs", flush=True)
    print(f"  GPU Latency:         {gpu_latency_ms:.2f} ms ({gpu_fps:.1f} FPS)", flush=True)
    print(f"  CPU Latency:         {cpu_latency_ms:.1f} ms", flush=True)
    print(f"  Training Time/Epoch: {sec_per_epoch:.1f} s", flush=True)
    print(f"  Total (No Tuning):   {train_time_no_tune_min:.1f} min ({train_time_no_tune_min/60:.2f} h)", flush=True)
    print(f"  Total (With Tuning): {train_time_with_tune_min:.1f} min ({train_time_with_tune_hours:.2f} h)", flush=True)

    # Clean up GPU memory
    del model, optimizer, train_batch, target_batch, dummy_input_cuda, dummy_input_cpu
    torch.cuda.empty_cache()
    gc.collect()

    return {
        'trainable_params_M': round(trainable_params / 1e6, 2),
        'gflops': round(gflops, 2),
        'gpu_latency_ms': round(gpu_latency_ms, 2),
        'gpu_fps': round(gpu_fps, 1),
        'cpu_latency_ms': round(cpu_latency_ms, 1),
        'sec_per_epoch': round(sec_per_epoch, 1),
        'train_time_no_tuning_min': round(train_time_no_tune_min, 1),
        'train_time_with_tuning_hours': round(train_time_with_tune_hours, 2)
    }

def main():
    model_factories = [
        ('Classic U-Net', lambda: get_unet()),
        ('UNet++', lambda: get_unetpp(deep_supervision=True)),
        ('RepSegNet V1', lambda: get_repsegnet()),
        ('RepSegNet V2 (Proposed)', lambda: get_repsegnet_v2(encoder_name=SEG_ENCODER_NAME, deep_supervision=True))
    ]

    results = {}
    for name, fn in model_factories:
        results[name] = benchmark_single_model(name, fn)

    out_file = 'outputs/computational_complexity_metrics.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n==========================================", flush=True)
    print(f"Saved complete computational metrics -> {out_file}", flush=True)
    print(f"==========================================", flush=True)

if __name__ == '__main__':
    main()
