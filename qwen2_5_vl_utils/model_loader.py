import torch
from transformers import AutoProcessor, AutoTokenizer

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError as exc:
    raise ImportError(
        "Qwen2.5-VL model class not available. Please upgrade `transformers` to a version that includes "
        "`Qwen2_5_VLForConditionalGeneration`."
    ) from exc


def load_qwen_model(model_name: str, gpu_id: int = 0):
    """
    Loads a Qwen2.5-VL model, tokenizer, and processor from Hugging Face.

    Args:
        model_name: Hugging Face repo id or local path.
        gpu_id: Target CUDA device index.

    Returns:
        tokenizer, processor, model, device (string)
    """
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return tokenizer, processor, model, device
