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

With MTEB wrapper:
Check [inference_mteb.py](inference_mteb.py)

With Transformers:
Check [inference_transformers.py](inference_transformers.py)

### MTEB Evaluation

```python
import mteb

model_meta = mteb.get_model_meta('eagerworks/eager-embed-v1')
model = model_meta.load_model()

# Get benchmarks and extract tasks from them
benchmarks = mteb.get_benchmarks(["ViDoRe(v2)"])
tasks = []
for benchmark in benchmarks:
    tasks.extend(benchmark.tasks)
print(tasks)
# Run evaluation with reduced batch size to save CUDA memory
results = mteb.evaluate(model=model, tasks=tasks, encode_kwargs={"batch_size": 8})

print("Evaluation complete!")
print(results)
```

For more examples, see [evaluate_mteb.py](evaluate_mteb.py)

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

