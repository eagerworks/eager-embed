import torch
from mteb_wrapper import EagerEmbedV1Wrapper


def example_inference():
    """Example of manual evaluation without MTEB tasks."""
    from PIL import Image
    import requests
    from io import BytesIO
    
    # Initialize wrapper
    wrapper = EagerEmbedV1Wrapper(
        model_name='eagerworks/eager-embed-v1',
        device='cuda:0',
        torch_dtype=torch.float16,
        image_size=784,
        use_peft=False,
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
    
    # Custom collate function for images
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
    print("\n" + "=" * 80)
    print("Example inference")
    print("=" * 80)
    example_inference()