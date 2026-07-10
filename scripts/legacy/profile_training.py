#!/usr/bin/env python3
"""
Training profiler utility for Guac models.

This script provides several methods to profile training performance:
1. PyTorch Profiler integration
2. Line profiler for hot path analysis
3. GPU utilization monitoring
4. Memory profiling
"""

import argparse
import torch
import time
from contextlib import contextmanager
from typing import Dict, Any


class TrainingProfiler:
    """Utility class for profiling training performance."""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.profiler = None

    @contextmanager
    def pytorch_profiler(self, output_dir="./profiling_results"):
        """Context manager for PyTorch profiler."""
        if not self.gpu_available:
            print("GPU not available, CPU-only profiling enabled")

        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.gpu_available:
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as profiler:
            self.profiler = profiler
            yield profiler

    def step(self):
        """Call this after each training step when using PyTorch profiler."""
        if self.profiler:
            self.profiler.step()

    def print_gpu_utilization(self):
        """Print current GPU utilization (requires nvidia-ml-py)."""
        if not self.gpu_available:
            print("GPU not available")
            return

        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU Utilization: {util.gpu}%")
            print(f"Memory Utilization: {util.memory}%")

            # Memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            print(f"GPU Memory: {memory_info.used / 1024**3:.1f} GB / {memory_info.total / 1024**3:.1f} GB")

        except ImportError:
            print("Install nvidia-ml-py for GPU utilization: pip install nvidia-ml-py")
        except Exception as e:
            print(f"Error getting GPU utilization: {e}")

    @staticmethod
    def profile_model_forward(model, sample_input, num_runs=100):
        """Profile just the forward pass of a model."""
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timing
        start_time = time.time()

        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(sample_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        avg_time = (end_time - start_time) / num_runs
        print(f"Average forward pass time: {avg_time*1000:.2f} ms")
        print(f"Forward passes per second: {1/avg_time:.1f}")

        return avg_time


def profile_with_line_profiler():
    """Instructions for using line_profiler."""
    print("""
    📊 LINE PROFILER USAGE:

    1. Install line_profiler: pip install line_profiler

    2. Add @profile decorator to functions you want to profile:
       @profile
       def train_step(model, batch):
           # Your training code here
           pass

    3. Run with kernprof:
       kernprof -l -v your_training_script.py

    This will show line-by-line timing information!
    """)


def profile_memory():
    """Instructions for memory profiling."""
    print("""
    💾 MEMORY PROFILING OPTIONS:

    1. PyTorch built-in memory profiler:
       torch.profiler.profile(profile_memory=True)

    2. Memory_profiler:
       pip install memory_profiler
       @profile  # decorator
       python -m memory_profiler your_script.py

    3. Tracemalloc (Python built-in):
       import tracemalloc
       tracemalloc.start()
       # your code
       current, peak = tracemalloc.get_traced_memory()

    4. GPU memory tracking:
       torch.cuda.memory_allocated()
       torch.cuda.memory_reserved()
       torch.cuda.max_memory_allocated()
    """)


def main():
    parser = argparse.ArgumentParser(description="Training profiling utilities")
    parser.add_argument('--mode', choices=['pytorch', 'line', 'memory', 'gpu'],
                       default='pytorch', help='Profiling mode')
    parser.add_argument('--output-dir', default='./profiling_results',
                       help='Output directory for profiling results')

    args = parser.parse_args()

    if args.mode == 'pytorch':
        print("""
        🔍 PYTORCH PROFILER USAGE:

        Add this to your training script:

        from scripts.training.profile_training import TrainingProfiler

        profiler = TrainingProfiler()
        with profiler.pytorch_profiler("./profiling_results"):
            for batch in dataloader:
                # Your training step
                loss = model(batch)
                loss.backward()
                optimizer.step()
                profiler.step()  # Important!

        Then view results with:
        tensorboard --logdir=./profiling_results
        """)

    elif args.mode == 'line':
        profile_with_line_profiler()

    elif args.mode == 'memory':
        profile_memory()

    elif args.mode == 'gpu':
        profiler = TrainingProfiler()
        profiler.print_gpu_utilization()


if __name__ == "__main__":
    main()