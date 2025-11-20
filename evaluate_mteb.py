"""
Example script showing how to use the Qwen3VL MTEB wrapper for evaluation.
"""

import torch
from mteb_wrapper import get_qwen3vl_model_meta

MODEL_NAME = "./run2_8x5090"


def example_mteb_evaluation():
    """Example of running MTEB evaluation."""
    import mteb
    
    # Create model meta
    model_meta = get_qwen3vl_model_meta(
        model_name=MODEL_NAME,
        revision="main",
        release_date="2024-11-01",
        n_parameters=3_000_000_000,
        memory_usage_mb=6000,
        embed_dim=2560,
        dtype=torch.float16,
        attn_implementation="flash_attention_2",
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
    model_meta = get_qwen3vl_model_meta(
        model_name=MODEL_NAME,
        revision="main",
        use_peft=True,
        image_size=784,
    )
    model_memory = model_meta.calculate_memory_usage_mb()
    print(f"Model memory usage: {model_memory} MB")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MTEB evaluation")
    print("=" * 80)
    example_mteb_evaluation()

