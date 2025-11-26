# Relevant code:
# https://github.com/texttron/tevatron/blob/main/examples/multimodal/qwen2.5vl/eval_vidore.py
# mteb/models/model_implementations/jina_models.py
# mteb/models/model_implementations/colqwen_models.py
# mteb/models/model_implementations/colpali_models.py

import logging
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mteb.abstasks.task_metadata import TaskMetadata
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ModelMeta, ScoringFunction
from mteb.types import Array, BatchedInput, PromptType

logger = logging.getLogger(__name__)


class EagerEmbedV1Wrapper(AbsEncoder):
    """Wrapper for EagerEmbed single-vector embedding models."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        image_size: int = 784,
        use_peft: bool = False,
        **kwargs,
    ):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.use_peft = use_peft

        # Load model
        if self.use_peft:
            from peft import PeftModel, PeftConfig

            config = PeftConfig.from_pretrained(model_name)
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.base_model_name_or_path, **kwargs
            )
            self.mdl = PeftModel.from_pretrained(base_model, model_name)
            self.mdl = self.mdl.merge_and_unload()
        else:
            self.mdl = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, **kwargs
            )

        self.mdl = self.mdl.to(self.device)
        self.mdl.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Make sure padding is on the left (check tokenizer_config.json)
        # self.processor.tokenizer.padding_side = "left"

    def get_embedding(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from last token of last hidden state."""
        reps = last_hidden_state[:, -1]
        return reps

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        """Encode inputs (text and/or images) into embeddings."""
        from qwen_vl_utils import process_vision_info

        all_embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in tqdm(inputs, desc="Encoding"):
                batch_texts = batch.get("text", [])
                batch_images = batch.get("image", [])

                messages = []
                for i in range(max(len(batch_texts), len(batch_images))):
                    text_content = batch_texts[i] if batch_texts else ""
                    image_content = batch_images[i] if batch_images else None

                    query_prefix = "Query: " if prompt_type == PromptType.query else ""
                    content = [
                        {"type": "text", "text": f"{query_prefix}{text_content}"}
                    ]

                    if image_content is not None:
                        content.append(
                            {
                                "type": "image",
                                "image": image_content,
                                "resized_height": self.image_size,
                                "resized_width": self.image_size,
                            }
                        )

                    messages.append([{"role": "user", "content": content}])

                # Prepare inputs
                texts = [
                    self.processor.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=False
                    )
                    + "<|endoftext|>"
                    for msg in messages
                ]

                image_inputs = None
                video_inputs = None
                if batch_images:
                    image_inputs, video_inputs = process_vision_info(messages)

                model_inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding="longest",
                    return_tensors="pt",
                ).to(self.device)

                # Get embeddings
                output = self.mdl(
                    **model_inputs, return_dict=True, output_hidden_states=True
                )
                embeddings = self.get_embedding(output.hidden_states[-1])
                embeddings = embeddings.cpu().to(torch.float32)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

                all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0)


EAGER_EMBED_V1_CITATION = """@article{EagerEmbed,
  title={Eager Embed V1: Multimodal Dense Embeddings for Retrieval},
  author={Juan Pablo Balarini},
  year={2025},
  publisher={Eagerworks},
  url={https://github.com/eagerworks/eager-embed},
}"""

EAGER_EMBED_V1_TRAINING_DATASETS = {"colpali", "bge-ir", "pixmo-docs", "wiki-ss"}


def get_eager_embed_v1_model_meta(
    model_name: str,
    revision: str | None = None,
    **loader_kwargs,
) -> ModelMeta:
    """
    Create a ModelMeta instance for a eager-embed-v1 embedding model.

    Args:
        model_name: HuggingFace model name or path
        revision: Model revision/commit hash
        n_parameters: Number of model parameters
        memory_usage_mb: Approximate memory usage in MB
        **loader_kwargs: Additional kwargs to pass to the loader
    """
    return ModelMeta(
        loader=EagerEmbedV1Wrapper,
        loader_kwargs=loader_kwargs,
        name=model_name,
        languages=["fra-Latn", "spa-Latn", "eng-Latn", "deu-Latn"],
        revision=revision,
        release_date="2025-11-20",
        modalities=["image", "text"],
        n_parameters=4_000_000_000,
        memory_usage_mb=16929,
        max_tokens=262144,
        embed_dim=2560,
        license="apache-2.0",
        open_weights=True,
        framework=["Tevatron"],
        reference="https://huggingface.co/eagerworks/eager-embed-v1",
        similarity_fn_name=ScoringFunction.COSINE,
        use_instructions=True,
        training_datasets=EAGER_EMBED_V1_TRAINING_DATASETS,
        citation=EAGER_EMBED_V1_CITATION,
        adapted_from="https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct",
        public_training_code="https://github.com/eagerworks/eager-embed",
        public_training_data="https://github.com/eagerworks/eager-embed/blob/main/dataset_config.yaml",
    )
