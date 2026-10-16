"""Stage plumbing shared by pretrain, midtrain and posttrain: resuming, checkpoints, early stopping."""

import pytest
import torch
from datasets import Dataset
from transformers import TrainerControl, TrainingArguments

from guac.core.custom_architectures import MixedLayerConfig, MixedLayerForCausalLM
from guac.core.dataset_utils import PackedSequenceCollator
from guac.core.pipeline_utils import GuacTrainer, ProgressEarlyStoppingCallback, load_tokenizer, run_training_loop


def tiny_trainer(output_dir, max_steps, tokenizer=None):
    torch.manual_seed(0)
    model = MixedLayerForCausalLM(MixedLayerConfig(vocab_size=23, hidden_size=32, num_attention_heads=4,
                                                   num_hidden_layers=1, layer_types=["conv"], intermediate_size=64))
    data = Dataset.from_dict({'input_ids': [[(i + j) % 23 for j in range(8)] for i in range(32)]})
    args = TrainingArguments(output_dir=str(output_dir), max_steps=max_steps, per_device_train_batch_size=4,
                             save_steps=2, logging_steps=1, report_to=[], use_cpu=True, seed=0)
    return GuacTrainer(model=model, args=args, train_dataset=data, data_collator=PackedSequenceCollator(),
                       processing_class=tokenizer)


def test_resuming_from_a_checkpoint(tmp_path):
    """transformers 4.46 reloads the RNG state with a torch.load that newer torch restricts."""
    straight = tiny_trainer(tmp_path / "straight", 4)
    run_training_loop(straight)
    tiny_trainer(tmp_path / "resumed", 2).train()
    resumed = tiny_trainer(tmp_path / "resumed", 4)
    run_training_loop(resumed, str(tmp_path / "resumed" / "checkpoint-2"))
    assert resumed.state.global_step == 4
    for a, b in zip(straight.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=0)


def test_checkpoints_hold_the_tokenizer(tmp_path):
    from guac.tokenizer.build_char_tokenizer import build_char_tokenizer
    tokenizer = build_char_tokenizer(str(tmp_path / "tok"))
    tiny_trainer(tmp_path / "run", 2, tokenizer).train()
    reloaded = load_tokenizer(str(tmp_path / "run" / "checkpoint-2"))
    assert reloaded("1 + 1")['input_ids'] == tokenizer("1 + 1")['input_ids']


def evaluate(callback, **metrics):
    control = TrainerControl()
    callback.on_evaluate(None, None, control, metrics=metrics)
    return control.should_training_stop


def test_early_stopping_waits_while_any_metric_improves():
    callback = ProgressEarlyStoppingCallback(2, threshold=0.01,
                                             metrics={'eval_loss': False, 'eval_heldout_accuracy': True})
    # Accuracy stuck at 0 while the loss keeps falling: keep training
    assert not any(evaluate(callback, eval_loss=loss, eval_heldout_accuracy=0.0) for loss in (3.0, 2.5, 2.0, 1.5))
    # Accuracy improving while the loss is flat: keep training
    assert not any(evaluate(callback, eval_loss=1.5, eval_heldout_accuracy=acc) for acc in (0.1, 0.2, 0.3))
    # Neither improves (by more than the threshold) for two evaluations: stop
    assert not evaluate(callback, eval_loss=1.495, eval_heldout_accuracy=0.3)
    assert evaluate(callback, eval_loss=1.5, eval_heldout_accuracy=0.305)


def test_checkpoint_at_every_evaluation_by_default(tmp_path):
    from guac.pipeline.midtrain import create_midtraining_arguments
    from guac.pipeline.posttrain import create_posttraining_arguments

    assert create_midtraining_arguments(str(tmp_path), eval_steps=30).save_steps == 30
    args = create_posttraining_arguments(str(tmp_path), 'sft', {'eval_steps': 300})
    assert (args.eval_steps, args.save_steps) == (300, 300)
    assert create_posttraining_arguments(str(tmp_path), 'instruction', {}).save_steps == 100
    with pytest.raises(ValueError, match="multiple"):
        create_posttraining_arguments(str(tmp_path), 'sft', {'eval_steps': 300, 'save_steps': 500})
