"""
Training callbacks for Guac arithmetic models.
"""

import time
import torch
import psutil
import os
from collections import defaultdict, deque
from transformers import TrainerCallback
from tqdm.auto import tqdm


class CustomProgressCallback(TrainerCallback):
    """
    Progress bar showing the latest loss, learning rate and gradient norm, plus
    a status line with a short greedy sample from the model every
    `sample_every` steps (0 turns sampling off).
    """

    def __init__(self, tokenizer, sample_every: int = 200, max_new_tokens: int = 32):
        self.tokenizer = tokenizer
        self.sample_every = sample_every
        self.max_new_tokens = max_new_tokens
        self.pbar = None
        self.status_bar = None
        self.last_train_log = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.status_bar = tqdm(total=1, bar_format='{desc}', position=0, leave=False)
        self.pbar = tqdm(total=state.max_steps, desc="Training", position=1, mininterval=0.5)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not (self.pbar and self.status_bar):
            return
        latest = next((entry for entry in reversed(state.log_history) if 'loss' in entry or 'train_loss' in entry), None)
        if latest is not None and latest is not self.last_train_log:
            tqdm.write(f"Training log: {latest}")
            loss = latest.get('loss', latest.get('train_loss'))
            parts = [f"Epoch:{state.epoch or 0:.2f}", f"Loss:{loss:.3f}"]
            if 'learning_rate' in latest:
                parts.append(f"LR:{latest['learning_rate']:.2e}")
            if 'grad_norm' in latest:
                parts.append(f"GradNorm:{latest['grad_norm']:.2f}")
            self.pbar.set_description_str("  ".join(parts) + "  ")
            self.last_train_log = latest
        self.pbar.update(1)

        if self.sample_every and model is not None and state.global_step % self.sample_every == 0:
            self.status_bar.set_description(self.sample(model))

    def sample(self, model) -> str:
        """Greedy continuation of the BOS token (EOS if there is none)."""
        # Not tokenizer(bos_token): most tokenizers append EOS, which would end the prompt
        start = self.tokenizer.bos_token_id
        if start is None:
            start = self.tokenizer.eos_token_id
        input_ids = torch.tensor([[start]], device=model.device)
        model.eval()
        with torch.no_grad():
            generated = model.generate(input_ids, attention_mask=torch.ones_like(input_ids),
                                       max_new_tokens=self.max_new_tokens, do_sample=False,
                                       pad_token_id=self.tokenizer.pad_token_id,
                                       eos_token_id=self.tokenizer.eos_token_id)[0]
        model.train()
        return self.tokenizer.decode(generated, skip_special_tokens=True).replace("\n", " ").strip()

    def on_train_end(self, args, state, control, **kwargs):
        if self.status_bar: self.status_bar.close()
        if self.pbar: self.pbar.close()


class ProfilingCallback(TrainerCallback):
    """
    Comprehensive profiling callback to track time spent in different training phases.

    Tracks:
    - Forward pass time
    - Backward pass time
    - Optimizer step time
    - Data loading time
    - GPU/CPU utilization
    - Memory usage
    """

    def __init__(self, log_every_n_steps=10, detailed_profiling=True):
        self.log_every_n_steps = log_every_n_steps
        self.detailed_profiling = detailed_profiling

        # Timing storage
        self.step_times = defaultdict(list)
        self.recent_times = defaultdict(lambda: deque(maxlen=100))

        # Phase tracking
        self.phase_start_time = None
        self.step_start_time = None
        self.forward_start_time = None
        self.backward_start_time = None

        # System monitoring
        self.process = psutil.Process(os.getpid())

        # GPU monitoring (if available)
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_device = torch.cuda.current_device()

    def _get_memory_info(self):
        """Get current memory usage info."""
        # CPU memory
        memory_info = self.process.memory_info()
        cpu_memory_mb = memory_info.rss / 1024 / 1024

        # GPU memory
        gpu_memory_mb = 0
        gpu_memory_cached_mb = 0
        if self.gpu_available:
            gpu_memory_bytes = torch.cuda.memory_allocated(self.gpu_device)
            gpu_memory_cached_bytes = torch.cuda.memory_reserved(self.gpu_device)
            gpu_memory_mb = gpu_memory_bytes / 1024 / 1024
            gpu_memory_cached_mb = gpu_memory_cached_bytes / 1024 / 1024

        return {
            'cpu_memory_mb': cpu_memory_mb,
            'gpu_memory_mb': gpu_memory_mb,
            'gpu_cached_mb': gpu_memory_cached_mb
        }

    def _log_timing(self, phase, duration):
        """Log timing for a specific phase."""
        self.step_times[phase].append(duration)
        self.recent_times[phase].append(duration)

    def on_step_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each training step."""
        self.step_start_time = time.time()

        if self.detailed_profiling:
            # Clear GPU cache for accurate memory measurement
            if self.gpu_available:
                torch.cuda.synchronize()

    def on_prediction_step(self, args, state, control, **kwargs):
        """Called before forward pass."""
        if self.detailed_profiling:
            if self.gpu_available:
                torch.cuda.synchronize()
            self.forward_start_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each training step."""
        if self.step_start_time is None:
            return

        # Calculate total step time
        if self.gpu_available:
            torch.cuda.synchronize()

        step_end_time = time.time()
        total_step_time = step_end_time - self.step_start_time
        self._log_timing('total_step', total_step_time)

        # Log periodically
        if state.global_step % self.log_every_n_steps == 0:
            self._print_profiling_stats(state.global_step)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs - can track optimizer and loss computation time."""
        if logs and self.detailed_profiling:
            # Add timing info to logs
            if self.recent_times['total_step']:
                avg_step_time = sum(self.recent_times['total_step']) / len(self.recent_times['total_step'])
                logs['avg_step_time'] = avg_step_time
                logs['steps_per_second'] = 1.0 / avg_step_time if avg_step_time > 0 else 0

            # Add memory info
            memory_info = self._get_memory_info()
            logs.update(memory_info)

    def _print_profiling_stats(self, step):
        """Print detailed profiling statistics."""
        print(f"\n📊 PROFILING STATS (Step {step}):")
        print("=" * 50)

        # Timing statistics
        for phase, times in self.recent_times.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                print(f"📈 {phase:15s}: {avg_time:6.3f}s avg ({min_time:.3f}-{max_time:.3f}s)")

        # Memory statistics
        memory_info = self._get_memory_info()
        print(f"💾 CPU Memory:     {memory_info['cpu_memory_mb']:6.1f} MB")
        if self.gpu_available:
            print(f"🎮 GPU Memory:     {memory_info['gpu_memory_mb']:6.1f} MB")
            print(f"🎮 GPU Cached:     {memory_info['gpu_cached_mb']:6.1f} MB")

        # Performance metrics
        if self.recent_times['total_step']:
            avg_step = sum(self.recent_times['total_step']) / len(self.recent_times['total_step'])
            steps_per_sec = 1.0 / avg_step if avg_step > 0 else 0
            print(f"⚡ Performance:    {steps_per_sec:.2f} steps/sec")

        print("=" * 50)

    def get_summary(self):
        """Get a summary of all profiling data."""
        summary = {}
        for phase, times in self.step_times.items():
            if times:
                summary[phase] = {
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'total': sum(times),
                    'count': len(times)
                }
        return summary


class LayerTimingCallback(TrainerCallback):
    """
    Detailed layer-by-layer timing callback for identifying bottleneck layers.

    This instruments the model to track time spent in each layer type.
    """

    def __init__(self, model, log_every_n_steps=50):
        self.model = model
        self.log_every_n_steps = log_every_n_steps
        self.layer_times = defaultdict(list)
        self.current_step = 0
        self.hooks = []

        # Install hooks on layers if model has layers attribute
        if hasattr(model, 'layers'):
            self._install_hooks()

    def _install_hooks(self):
        """Install timing hooks on each layer."""
        def make_timing_hook(layer_name, layer_type):
            def timing_hook(module, input, output):
                if hasattr(module, '_timing_start'):
                    duration = time.time() - module._timing_start
                    self.layer_times[f"{layer_type}_{layer_name}"].append(duration)

            def pre_hook(module, input):
                module._timing_start = time.time()

            return pre_hook, timing_hook

        # Hook each layer
        for i, layer in enumerate(self.model.layers):
            layer_type = getattr(layer, 'layer_type', f'layer_{i}')
            pre_hook, post_hook = make_timing_hook(i, layer_type)

            # Register hooks
            handle1 = layer.register_forward_pre_hook(pre_hook)
            handle2 = layer.register_forward_hook(post_hook)
            self.hooks.extend([handle1, handle2])

    def on_step_end(self, args, state, control, **kwargs):
        """Log layer timing periodically."""
        self.current_step += 1

        if self.current_step % self.log_every_n_steps == 0:
            self._print_layer_stats()

    def _print_layer_stats(self):
        """Print layer-by-layer timing statistics."""
        print(f"\n🔍 LAYER TIMING ANALYSIS:")
        print("=" * 60)

        # Group by layer type
        layer_type_times = defaultdict(list)
        for layer_name, times in self.layer_times.items():
            if times:
                layer_type = layer_name.split('_')[0]
                avg_time = sum(times[-10:]) / min(len(times), 10)  # Last 10 steps
                layer_type_times[layer_type].append(avg_time)

        # Print aggregated stats by layer type
        for layer_type, times in layer_type_times.items():
            if times:
                total_time = sum(times)
                avg_time = total_time / len(times)
                print(f"🧬 {layer_type:15s}: {total_time:6.3f}s total, {avg_time:6.3f}s avg/layer")

        # Print individual layer stats
        print("\nDetailed layer breakdown:")
        for layer_name, times in sorted(self.layer_times.items()):
            if times:
                recent_times = times[-10:]  # Last 10 steps
                avg_time = sum(recent_times) / len(recent_times)
                print(f"  {layer_name:20s}: {avg_time:6.3f}s")

        print("=" * 60)

    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()