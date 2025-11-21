import torch
from transformers import AutoProcessor, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError as exc:
    raise ImportError(
        "AutoModelForImageTextToText not available. Please upgrade `transformers` to a version >= 4.57.0."
    ) from exc


def load_qwen3_model(model_name: str, gpu_id: int = 0):
    """
    Loads a Qwen3-VL model, tokenizer, and processor from Hugging Face.

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
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return tokenizer, processor, model, device
