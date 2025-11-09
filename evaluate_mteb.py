"""
Example script showing how to use the Qwen3VL MTEB wrapper for evaluation.
"""

import torch
from mteb_wrapper import Qwen3VLEmbeddingWrapper, get_qwen3vl_model_meta

# Method 2: Using with MTEB tasks
def example_mteb_evaluation():
    """Example of running MTEB evaluation."""
    import mteb
    
    # Create model meta
    model_meta = get_qwen3vl_model_meta(
        model_name="./retriever-qwen3vl-colpali-100k",
        revision="main",
        release_date="2024-11-01",
        n_parameters=3_000_000_000,
        memory_usage_mb=6000,
        embed_dim=2560,
        torch_dtype=torch.float16,
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


# Method 3: Manual evaluation on custom data
def example_manual_evaluation():
    """Example of manual evaluation without MTEB tasks."""
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Initialize wrapper
    wrapper = Qwen3VLEmbeddingWrapper(
        model_name='retriever-qwen3vl-colpali-100k',
        device='cuda:0',
        torch_dtype=torch.float16,
        image_size=784,
        use_peft=True,
    )
    
    # Prepare some test queries
    queries = [
        "Where can we find the animal llama?",
        "What is the LLaMA AI model?",
    ]
    
    # Prepare some test images
    urls = [
        "https://huggingface.co/Tevatron/dse-phi3-docmatix-v2/resolve/main/animal-llama.png",
        "https://huggingface.co/Tevatron/dse-phi3-docmatix-v2/resolve/main/meta-llama.png",
    ]
    
    headers = {'User-Agent': 'MTEB Evaluation 1.0'}
    images = [
        Image.open(BytesIO(requests.get(url, headers=headers).content))
        for url in urls
    ]
    
    # Create simple datasets (normally you'd use proper DataLoader)
    from datasets import Dataset
    from torch.utils.data import DataLoader
    
    query_dataset = Dataset.from_dict({"text": queries})
    image_dataset = Dataset.from_dict({"image": images})
    
    # Custom collate function for PIL images
    def collate_fn(batch):
        return {"image": [item["image"] for item in batch]}
    
    query_loader = DataLoader(query_dataset, batch_size=2)
    image_loader = DataLoader(image_dataset, batch_size=2, collate_fn=collate_fn)
    
    # Get embeddings
    query_embeddings = wrapper.get_text_embeddings(query_loader)
    image_embeddings = wrapper.get_image_embeddings(image_loader)
    
    # Calculate similarities
    similarities = wrapper.similarity(query_embeddings, image_embeddings)
    
    print("Queries:", queries)
    print("Images:", urls)
    print("Similarity matrix:")
    print(similarities)
    
    return similarities


if __name__ == "__main__":
    # print("\n" + "=" * 80)
    # print("Example 3: Manual evaluation")
    # print("=" * 80)
    # example_manual_evaluation()
    
    print("\n" + "=" * 80)
    print("Example 2: MTEB evaluation")
    print("=" * 80)
    example_mteb_evaluation()

