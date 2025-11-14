import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def load_qwen_model(model_name: str, gpu_id: int = 0):
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return tokenizer, processor, model, device
