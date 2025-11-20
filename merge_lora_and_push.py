import argparse
import logging
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_and_push(adapter_path, push_to_hub_id, base_model_path=None, dtype="float16", token=None, device="auto"):
    logger.info(f"Loading adapter config from {adapter_path}")
    config = PeftConfig.from_pretrained(adapter_path)
    
    base_model_name = base_model_path or config.base_model_name_or_path
    logger.info(f"Base model: {base_model_name}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32
    }
    torch_dtype = dtype_map.get(dtype, torch.float16)

    logger.info("Loading base model...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_name,
        dtype=torch_dtype,
        device_map=device,
        trust_remote_code=True
    )

    logger.info("Loading PEFT model...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    logger.info("Merging and unloading...")
    model = model.merge_and_unload()

    logger.info("Loading processor...")
    # Try loading processor from adapter path first to capture any tokenizer changes
    try:
        processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
        logger.info(f"Loaded processor from {adapter_path}")
    except Exception as e:
        logger.warning(f"Could not load processor from adapter path: {e}. Falling back to base model.")
        processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
        logger.info(f"Loaded processor from {base_model_name}")

    logger.info(f"Pushing to hub: {push_to_hub_id}")
    
    model.push_to_hub(push_to_hub_id, token=token)
    processor.push_to_hub(push_to_hub_id, token=token)
    
    logger.info("Successfully pushed model and processor to hub.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter and push to Hub")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--push_to_hub_id", type=str, required=True, help="Target Hub repository ID")
    parser.add_argument("--base_model_path", type=str, default=None, help="Override base model path")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"], help="Model data type")
    parser.add_argument("--token", type=str, default=None, help="HF API token")
    parser.add_argument("--device", type=str, default="auto", help="Device map (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    merge_and_push(
        adapter_path=args.adapter_path,
        push_to_hub_id=args.push_to_hub_id,
        base_model_path=args.base_model_path,
        dtype=args.dtype,
        token=args.token,
        device=args.device
    )

