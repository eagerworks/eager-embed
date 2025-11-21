# Eager Embed V1

A multimodal embedding model based on Qwen3-VL for text and image retrieval tasks. Built with [Tevatron](https://github.com/texttron/tevatron) framework and fine-tuned using LoRA.

## Features

- **Multimodal retrieval**: Encode text queries and images into a shared embedding space
- **Based on Qwen3-VL-4B-Instruct**: Leverages powerful vision-language foundation model
- **Efficient training**: Uses LoRA for parameter-efficient fine-tuning
- **MTEB evaluation**: Compatible with MTEB benchmark tasks

## Installation

```bash
# Install with uv
uv sync
```

## Training

Train a retriever model using the provided training script:

```bash
bash train.sh
```

The training uses:
- DeepSpeed for distributed training
- LoRA fine-tuning of Qwen3-VL-4B-Instruct
- Contrastive learning with temperature scaling
- EOS pooling with normalized embeddings

Configure your training data in `dataset_config.yaml`.

## Usage

### Basic Embedding Extraction

```python
import torch
from mteb_wrapper import Qwen3VLEmbeddingWrapper
from PIL import Image
from datasets import Dataset
from torch.utils.data import DataLoader

# Initialize the model
model = Qwen3VLEmbeddingWrapper(
    model_name='retriever-qwen3vl-colpali',
    dtype=torch.float16,
    image_size=784,
    use_peft=True,
)

# Encode text queries
queries = ["What is a llama?", "Show me mountain landscapes"]
query_dataset = Dataset.from_dict({"text": queries})
query_loader = DataLoader(query_dataset, batch_size=2)
query_embeddings = model.get_text_embeddings(query_loader)

# Encode images
images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
image_dataset = Dataset.from_dict({"image": images})
image_loader = DataLoader(
    image_dataset, 
    batch_size=2,
    collate_fn=lambda batch: {"image": [item["image"] for item in batch]}
)
image_embeddings = model.get_image_embeddings(image_loader)

# Calculate similarities
similarities = model.similarity(query_embeddings, image_embeddings)
print(similarities)
```

### MTEB Evaluation

```python
import torch
import mteb
from mteb_wrapper import get_qwen3vl_model_meta

# Create model metadata
model_meta = get_qwen3vl_model_meta(
    model_name="./retriever-qwen3vl-colpali",
    revision="main",
    release_date="2024-11-01",
    n_parameters=3_000_000_000,
    memory_usage_mb=6000,
    embed_dim=2560,
    dtype=torch.float16,
    use_peft=True,
    image_size=784,
)

# Load and evaluate
model = model_meta.load_model()
tasks = mteb.get_benchmark("ViDoRe(v2)")
results = mteb.evaluate(
    model=model, 
    tasks=tasks, 
    encode_kwargs={"batch_size": 8}
)
print(results)
```

For more examples, see `evaluate_mteb.py`.

### Merge LORA to base model and push to HF

```python
python merge_lora_and_push.py \
    --adapter_path run2_8x5090 \
    --push_to_hub_id eagerworks/eager-embed-v1 \
    --dtype float32
```

## Model Architecture

- **Base Model**: Qwen3-VL-4B-Instruct (~3B parameters)
- **Embedding Dimension**: 2560
- **Pooling**: Last token (EOS) pooling
- **Normalization**: L2 normalized embeddings
- **Training**: Contrastive learning with in-batch negatives

## License

Apache 2.0

## Citation

```bibtex
@article{EagerEmbed,
  title={Eager Embed V1: Multimodal Dense Embeddings for Retrieval},
  author={Juan Pablo Balarini},
  year={2025},
  publisher={Eagerworks},
  url={https://github.com/eagerworks/eager-embed}
}
```

