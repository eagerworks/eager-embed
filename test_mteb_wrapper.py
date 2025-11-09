"""
Simple test script to verify the MTEB wrapper works correctly.
"""

import torch
from mteb_wrapper import Qwen3VLEmbeddingWrapper
from PIL import Image
import requests
from io import BytesIO


def test_wrapper_initialization():
    """Test that the wrapper initializes correctly."""
    print("Testing wrapper initialization...")
    
    try:
        wrapper = Qwen3VLEmbeddingWrapper(
            model_name='retriever-qwen3vl-colpali',
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            torch_dtype=torch.float16,
            image_size=784,
            use_peft=True,
        )
        print("✓ Wrapper initialized successfully")
        return wrapper
    except Exception as e:
        print(f"✗ Failed to initialize wrapper: {e}")
        raise


def test_text_embeddings(wrapper):
    """Test text (query) embedding generation."""
    print("\nTesting text embeddings...")
    
    try:
        from datasets import Dataset
        from torch.utils.data import DataLoader
        
        queries = [
            "Where can we find the animal llama?",
            "What is the LLaMA AI model?",
            "Photo of a dog"
        ]
        
        query_dataset = Dataset.from_dict({"text": queries})
        query_loader = DataLoader(query_dataset, batch_size=2)
        
        embeddings = wrapper.get_text_embeddings(query_loader)
        
        assert embeddings.shape[0] == len(queries), "Number of embeddings doesn't match number of queries"
        assert len(embeddings.shape) == 2, "Embeddings should be 2D (batch, dim)"
        
        # Check normalization (allow for float16->float32 conversion tolerance)
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(
            norms, 
            torch.ones(len(queries)), 
            atol=1e-3
        ), f"Embeddings should be normalized, got norms: {norms}"
        
        print(f"✓ Text embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
        print(f"  Sample embedding norm: {torch.norm(embeddings[0]).item():.6f}")
        return embeddings
        
    except Exception as e:
        print(f"✗ Failed to generate text embeddings: {e}")
        raise


def test_image_embeddings(wrapper):
    """Test image (document) embedding generation."""
    print("\nTesting image embeddings...")
    
    try:
        from datasets import Dataset
        from torch.utils.data import DataLoader
        
        # Load test images
        urls = [
            "https://huggingface.co/Tevatron/dse-phi3-docmatix-v2/resolve/main/animal-llama.png",
            "https://huggingface.co/Tevatron/dse-phi3-docmatix-v2/resolve/main/meta-llama.png",
        ]
        
        headers = {'User-Agent': 'MTEB Test 1.0'}
        images = []
        for url in urls:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                images.append(Image.open(BytesIO(response.content)))
            except Exception as e:
                print(f"Warning: Failed to load image from {url}: {e}")
                print("Using a dummy image instead...")
                images.append(Image.new('RGB', (224, 224), color='red'))
        
        image_dataset = Dataset.from_dict({"image": images})
        
        # Custom collate function for PIL images
        def collate_fn(batch):
            return {"image": [item["image"] for item in batch]}
        
        image_loader = DataLoader(image_dataset, batch_size=2, collate_fn=collate_fn)
        
        embeddings = wrapper.get_image_embeddings(image_loader)
        
        assert embeddings.shape[0] == len(images), "Number of embeddings doesn't match number of images"
        assert len(embeddings.shape) == 2, "Embeddings should be 2D (batch, dim)"
        
        # Check normalization (allow for float16->float32 conversion tolerance)
        norms = torch.norm(embeddings, dim=1)
        assert torch.allclose(
            norms, 
            torch.ones(len(images)), 
            atol=1e-3
        ), f"Embeddings should be normalized, got norms: {norms}"
        
        print(f"✓ Image embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
        print(f"  Sample embedding norm: {torch.norm(embeddings[0]).item():.6f}")
        return embeddings
        
    except Exception as e:
        print(f"✗ Failed to generate image embeddings: {e}")
        raise


def test_similarity_computation(query_embeddings, image_embeddings):
    """Test similarity computation between queries and images."""
    print("\nTesting similarity computation...")
    
    try:
        similarities = torch.nn.functional.cosine_similarity(
            query_embeddings.unsqueeze(1),
            image_embeddings.unsqueeze(0),
            dim=-1
        )
        
        assert similarities.shape == (query_embeddings.shape[0], image_embeddings.shape[0]), \
            "Similarity matrix has wrong shape"
        assert torch.all(similarities >= -1.0) and torch.all(similarities <= 1.0), \
            "Cosine similarities should be in [-1, 1]"
        
        print(f"✓ Similarity computation: shape={similarities.shape}")
        print(f"  Similarity matrix:")
        print(similarities.numpy())
        
        # Print top match for each query
        for i in range(similarities.shape[0]):
            top_idx = similarities[i].argmax().item()
            top_score = similarities[i, top_idx].item()
            print(f"  Query {i} -> Image {top_idx} (score: {top_score:.4f})")
        
        return similarities
        
    except Exception as e:
        print(f"✗ Failed to compute similarities: {e}")
        raise


def test_embedding_dimensions(wrapper, query_embeddings, image_embeddings):
    """Test that embedding dimensions match expected values."""
    print("\nTesting embedding dimensions...")
    
    try:
        assert query_embeddings.shape[1] == image_embeddings.shape[1], \
            "Query and image embeddings should have the same dimension"
        
        embed_dim = query_embeddings.shape[1]
        print(f"✓ Embedding dimension: {embed_dim}")
        
        return embed_dim
        
    except Exception as e:
        print(f"✗ Failed dimension test: {e}")
        raise


def main():
    """Run all tests."""
    print("=" * 80)
    print("MTEB Wrapper Test Suite")
    print("=" * 80)
    
    try:
        # Test 1: Initialization
        wrapper = test_wrapper_initialization()
        
        # Test 2: Text embeddings
        query_embeddings = test_text_embeddings(wrapper)
        
        # Test 3: Image embeddings
        image_embeddings = test_image_embeddings(wrapper)
        
        # Test 4: Embedding dimensions
        embed_dim = test_embedding_dimensions(wrapper, query_embeddings, image_embeddings)
        
        # Test 5: Similarity computation
        similarities = test_similarity_computation(query_embeddings, image_embeddings)
        
        print("\n" + "=" * 80)
        print("All tests passed! ✓")
        print("=" * 80)
        print(f"\nModel Summary:")
        print(f"  - Device: {wrapper.device}")
        print(f"  - Embedding dimension: {embed_dim}")
        print(f"  - Image size: {wrapper.image_size}")
        print(f"  - Using PEFT: {wrapper.use_peft}")
        
        return True
        
    except Exception as e:
        print("\n" + "=" * 80)
        print(f"Tests failed: {e}")
        print("=" * 80)
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

