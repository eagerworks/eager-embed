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


class Qwen3VLEmbeddingWrapper(AbsEncoder):
    """Wrapper for Qwen3VL single-vector embedding models."""

    def __init__(
        self,
        model_name: str,
        revision: str | None = None,
        device: str | None = None,
        image_size: int = 784,
        use_peft: bool = True,
        **kwargs,
    ):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.image_size = image_size
        self.use_peft = use_peft
        
        # Handle deprecated torch_dtype parameter
        if 'torch_dtype' in kwargs:
            kwargs['dtype'] = kwargs.pop('torch_dtype')
        
        # Load model
        if self.use_peft:
            from peft import PeftModel, PeftConfig
            config = PeftConfig.from_pretrained(model_name)
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                config.base_model_name_or_path,
                **kwargs
            )
            self.mdl = PeftModel.from_pretrained(base_model, model_name)
            self.mdl = self.mdl.merge_and_unload()
        else:
            self.mdl = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                **kwargs
            )
        
        self.mdl = self.mdl.to(self.device)
        self.mdl.eval()
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)

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
        text_embeddings = None
        image_embeddings = None

        if "text" in inputs.dataset.features:
            text_embeddings = self.get_text_embeddings(inputs, **kwargs)

        if "image" in inputs.dataset.features:
            image_embeddings = self.get_image_embeddings(inputs, **kwargs)

        if text_embeddings is not None and image_embeddings is not None:
            if len(text_embeddings) != len(image_embeddings):
                raise ValueError(
                    "The number of texts and images must have the same length"
                )
            # For multimodal inputs, concatenate or fuse embeddings
            fused_embeddings = text_embeddings + image_embeddings
            return fused_embeddings
        elif text_embeddings is not None:
            return text_embeddings
        elif image_embeddings is not None:
            return image_embeddings
        
        raise ValueError("No text or image inputs found")

    def get_image_embeddings(
        self,
        images,
        batch_size: int = 32,
        **kwargs,
    ):
        """Encode images (documents) into embeddings."""
        from qwen_vl_utils import process_vision_info
        import torchvision.transforms.functional as F

        all_embeds = []
        
        # Create a new DataLoader with custom collate function to handle images
        def image_collate_fn(batch):
            """Custom collate function that keeps images as a list."""
            collated = {}
            for key in batch[0]:
                collated[key] = [item[key] for item in batch]
            return collated
        
        # Extract the dataset from the DataLoader and create a new one with proper collation
        dataset = images.dataset
        image_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=image_collate_fn,
            shuffle=False,
        )
        
        with torch.no_grad():
            for batch in tqdm(image_loader, desc="Encoding images"):
                # Convert batch to PIL images if needed
                imgs = [
                    F.to_pil_image(b.to(self.device))
                    if not isinstance(b, Image.Image)
                    else b
                    for b in batch["image"]
                ]
                
                # Create messages for each image
                doc_messages = []
                for img in imgs:
                    message = [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': ''},
                                {
                                    'type': 'image',
                                    'image': img,
                                    'resized_height': self.image_size,
                                    'resized_width': self.image_size
                                }
                            ]
                        }
                    ]
                    doc_messages.append(message)
                
                # Prepare inputs
                doc_texts = [
                    self.processor.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=False
                    ) + "<|endoftext|>"
                    for msg in doc_messages
                ]
                
                doc_image_inputs, doc_video_inputs = process_vision_info(doc_messages)
                doc_inputs = self.processor(
                    text=doc_texts,
                    images=doc_image_inputs,
                    videos=doc_video_inputs,
                    padding='longest',
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                output = self.mdl(**doc_inputs, return_dict=True, output_hidden_states=True)
                embeddings = self.get_embedding(output.hidden_states[-1])
                # Convert to float32 and ensure normalization is maintained
                embeddings = embeddings.cpu().to(torch.float32)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                all_embeds.append(embeddings)
        
        # Concatenate all embeddings
        all_embeds = torch.cat(all_embeds, dim=0)
        return all_embeds

    def get_text_embeddings(
        self,
        texts,
        batch_size: int = 32,
        **kwargs,
    ):
        """Encode texts (queries) into embeddings."""
        from qwen_vl_utils import process_vision_info

        all_embeds = []
        
        # Create a new DataLoader with custom collate function to handle variable-length texts
        def text_collate_fn(batch):
            """Custom collate function that doesn't try to stack text strings."""
            collated = {}
            for key in batch[0]:
                collated[key] = [item[key] for item in batch]
            return collated
        
        # Extract the dataset from the DataLoader and create a new one with proper collation
        dataset = texts.dataset
        text_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=text_collate_fn,
            shuffle=False,
        )
        
        with torch.no_grad():
            for batch in tqdm(text_loader, desc="Encoding texts"):
                # Create query messages
                query_messages = []
                for query in batch["text"]:
                    message = [
                        {
                            'role': 'user',
                            'content': [
                                {'type': 'text', 'text': f'Query: {query}'},
                            ]
                        }
                    ]
                    query_messages.append(message)
                
                # Prepare inputs
                query_texts = [
                    self.processor.apply_chat_template(
                        msg, tokenize=False, add_generation_prompt=False
                    ) + "<|endoftext|>"
                    for msg in query_messages
                ]
                
                query_image_inputs, query_video_inputs = process_vision_info(query_messages)
                query_inputs = self.processor(
                    text=query_texts,
                    images=query_image_inputs,
                    videos=query_video_inputs,
                    padding='longest',
                    return_tensors='pt'
                ).to(self.device)
                
                # Get embeddings
                output = self.mdl(**query_inputs, return_dict=True, output_hidden_states=True)
                embeddings = self.get_embedding(output.hidden_states[-1])
                # Convert to float32 and ensure normalization is maintained
                embeddings = embeddings.cpu().to(torch.float32)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
                all_embeds.append(embeddings)
        
        # Concatenate all embeddings
        all_embeds = torch.cat(all_embeds, dim=0)
        return all_embeds

    def get_fused_embeddings(
        self,
        texts: list[str] | None = None,
        images: list[Image.Image] | DataLoader | None = None,
        *,
        task_name: str | None = None,
        prompt_type: PromptType | None = None,
        batch_size: int = 32,
        fusion_mode="sum",
        **kwargs: Any,
    ):
        raise NotImplementedError(
            "Fused embeddings are not supported yet. Please use get_text_embeddings or get_image_embeddings."
        )

    def calculate_probs(self, text_embeddings, image_embeddings):
        """Calculate probabilities using softmax over cosine similarities."""
        scores = torch.nn.functional.cosine_similarity(
            text_embeddings.unsqueeze(1),
            image_embeddings.unsqueeze(0),
            dim=-1
        )
        return scores.softmax(dim=-1)

    def similarity(self, a, b):
        """Calculate cosine similarity between embeddings."""
        return torch.nn.functional.cosine_similarity(
            a.unsqueeze(1),
            b.unsqueeze(0),
            dim=-1
        )


QWEN3VL_CITATION = """
@article{Qwen3-VL,
  title={Qwen3-VL: Efficient Vision-Language Models},
  author={Qwen Team},
  year={2024}
}
"""


def get_qwen3vl_model_meta(
    model_name: str,
    revision: str | None = None,
    release_date: str = "2024-11-01",
    n_parameters: int = 3_000_000_000,
    memory_usage_mb: int = 6000,
    embed_dim: int = 2560,
    training_datasets: set[str] | None = None,
    **loader_kwargs,
) -> ModelMeta:
    """
    Create a ModelMeta instance for a Qwen3VL embedding model.
    
    Args:
        model_name: HuggingFace model name or path
        revision: Model revision/commit hash
        release_date: Release date in YYYY-MM-DD format
        n_parameters: Number of model parameters
        memory_usage_mb: Approximate memory usage in MB
        embed_dim: Embedding dimension
        training_datasets: Set of training dataset names
        **loader_kwargs: Additional kwargs to pass to the loader
    """
    return ModelMeta(
        loader=Qwen3VLEmbeddingWrapper,
        loader_kwargs=loader_kwargs,
        name=model_name,
        languages=["eng-Latn"],  # Add more languages if applicable
        revision=revision,
        release_date=release_date,
        modalities=["image", "text"],
        n_parameters=n_parameters,
        memory_usage_mb=memory_usage_mb,
        max_tokens=8192,  # Adjust based on your model
        embed_dim=embed_dim,
        license="apache-2.0",
        open_weights=True,
        framework=["Tevatron"],
        reference=f"https://huggingface.co/{model_name}",
        similarity_fn_name=ScoringFunction.COSINE,
        use_instructions=True,
        training_datasets=training_datasets or set(),
        citation=QWEN3VL_CITATION,
        public_training_code="https://github.com/illuin-tech/colpali",
        public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set"
    )


# Example usage:
# qwen3vl_colpali_100k = get_qwen3vl_model_meta(
#     model_name="your-username/retriever-qwen3vl-colpali-100k",
#     revision="main",
#     release_date="2024-11-01",
#     n_parameters=3_000_000_000,
#     memory_usage_mb=6000,
#     embed_dim=2560,
#     torch_dtype=torch.float16,
#     use_peft=True,
#     image_size=784,
# )

