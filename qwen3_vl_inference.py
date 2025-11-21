import argparse
import json
import os

from PIL import Image
import torch

from qwen3_vl_utils.model_loader import load_qwen3_model
from qwen3_vl_utils import prompt_wrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on Qwen3-VL with a single image.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Hugging Face identifier or path for the Qwen3-VL checkpoint.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id to load the model on.")
    parser.add_argument("--image-file", type=str, required=True, help="Path to the image for inference.")
    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.",
                        help="System message prepended to the chat template.")
    parser.add_argument("--user-prompt", type=str, default="",
                        help="Optional user text accompanying the image.")
    parser.add_argument("--output-file", type=str, default="qwen3_inference.json",
                        help="Where to store the inference result in JSON format.")
    return parser.parse_args()


def load_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    return Image.open(path).convert("RGB")


def count_images(messages):
    count = 0
    for turn in messages:
        content = turn.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                count += 1
    return count


def main():
    args = parse_args()

    tokenizer, processor, model, device = load_qwen3_model(args.model_name, args.gpu_id)
    base_messages = prompt_wrapper.build_base_messages(args.system_prompt, args.user_prompt)

    image = load_image(args.image_file)
    messages = prompt_wrapper.append_assistant_response(base_messages, "")
    num_images = count_images(messages)
    image_placeholders = [None] * num_images if num_images > 0 else None
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        images=image_placeholders,
    )

    inputs = processor(
        text=[prompt_text],
        images=[image],
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            temperature=0.2,
            max_new_tokens=512,
            use_cache=True,
        )

    prompt_len = inputs["input_ids"].shape[1]
    response = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()

    print("Model response:\n", response)

    result = {
        "model": args.model_name,
        "image": args.image_file,
        "system_prompt": args.system_prompt,
        "user_prompt": args.user_prompt,
        "response": response,
    }

    with open(args.output_file, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
