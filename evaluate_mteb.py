"""
Example script showing how to use the Qwen3VL MTEB wrapper for evaluation.
"""

import mteb
from mteb_wrapper import get_qwen3vl_model_meta


def evaluate_mteb():
    """MTEB evaluation with uploaded model."""

    model_meta = mteb.get_model_meta('eagerworks/eager-embed-v1')
    model = model_meta.load_model()

    tasks = mteb.get_benchmark("ViDoRe(v3)")
    print(tasks)
    # Run evaluation with reduced batch size to save CUDA memory
    results = mteb.evaluate(model=model, tasks=tasks, encode_kwargs={"batch_size": 8})

    print("Evaluation complete!")
    print(results)


def evaluate_mteb_with_custom_model():
    """MTEB evaluation with local model."""
    import torch

    # Create model meta
    model_meta = get_qwen3vl_model_meta(
        model_name="./run2_8x5090",
        revision="main",
        release_date="2024-11-01",
        n_parameters=3_000_000_000,
        memory_usage_mb=6000,
        embed_dim=2560,
        dtype=torch.float16,
        # attn_implementation="flash_attention_2", # Uncomment this for faster inference
        use_peft=True,
        image_size=784,
    )

    # Initialize wrapper
    model = model_meta.load_model()
    tasks = mteb.get_benchmark("ViDoRe(v2)")
    print(tasks)
    # Run evaluation with reduced batch size to save CUDA memory
    results = mteb.evaluate(model=model, tasks=tasks, encode_kwargs={"batch_size": 8})

    print("Evaluation complete!")
    print(results)


def compute_memory_usage():
    import mteb

    print("Computing memory usage...")
    model_meta = mteb.get_model_meta('eagerworks/eager-embed-v1')
    model_memory = model_meta.calculate_memory_usage_mb()
    print(f"Model memory usage: {model_memory} MB")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MTEB evaluation")
    print("=" * 80)
    evaluate_mteb()

