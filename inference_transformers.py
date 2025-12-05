import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.utils.import_utils import is_flash_attn_2_available
from qwen_vl_utils import process_vision_info

MODEL_NAME = "eagerworks/eager-embed-v1"
DEVICE = torch.device("cpu")
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
DTYPE = torch.bfloat16

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    attn_implementation=(
        "flash_attention_2" if is_flash_attn_2_available() else None
    ),
    dtype=DTYPE
).to(DEVICE).eval()

# Function to Encode Message
def encode_message(message):
    with torch.no_grad():
        texts = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
        image_inputs, video_inputs = process_vision_info(message)

        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding="longest",
        ).to(DEVICE)

        model_outputs = model(**inputs, return_dict=True, output_hidden_states=True)

        last_hidden_state = model_outputs.hidden_states[-1]
        embeddings = last_hidden_state[:, -1]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return embeddings


# Image Document Retrieval (Image, Chart, PDF)
MAX_IMAGE_SIZE = 784
example_query = 'Query: Where can we find the animal llama?'
example_image_1 = "https://huggingface.co/Tevatron/dse-phi3-docmatix-v2/resolve/main/animal-llama.png"
example_image_2 = "https://huggingface.co/Tevatron/dse-phi3-docmatix-v2/resolve/main/meta-llama.png"
query = [{'role': 'user', 'content': [{'type': 'text', 'text': example_query}]}]
image_1 = [{'role': 'user', 'content': [{'type': 'image', 'image': example_image_1, 'resized_height': MAX_IMAGE_SIZE, 'resized_width': MAX_IMAGE_SIZE}]}]
image_2 = [{'role': 'user', 'content': [{'type': 'image', 'image': example_image_2, 'resized_height': MAX_IMAGE_SIZE, 'resized_width': MAX_IMAGE_SIZE}]}]

sim1 = torch.cosine_similarity(encode_message(query), encode_message(image_1))
sim2 = torch.cosine_similarity(encode_message(query), encode_message(image_2))

print("Similarities:", sim1.item(), sim2.item())