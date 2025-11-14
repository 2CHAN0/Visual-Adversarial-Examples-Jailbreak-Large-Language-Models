import argparse
import csv
import os

import torch
from PIL import Image
from torchvision.utils import save_image
from transformers import AutoModelForVision2Seq, AutoProcessor

from qwen3_vl_utils import PromptWrapper, Attacker, DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(description="Visual adversarial attack on Qwen3-VL.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct", help="Hugging Face model id.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU to load the model on.")
    parser.add_argument("--n-iters", type=int, default=5000, help="Number of iterations for the attack.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of targets sampled per iteration.")
    parser.add_argument("--eps", type=int, default=32, help="Epsilon of the attack budget (in 1/255 units).")
    parser.add_argument("--alpha", type=int, default=1, help="Step size of the attack (in 1/255 units).")
    parser.add_argument("--constrained", action="store_true", help="Use an L-infinity constrained attack.")
    parser.add_argument("--save-dir", type=str, default="output", help="Directory to store adversarial images.")
    parser.add_argument("--template-image", type=str, default="adversarial_images/clean.jpeg", help="Path to seed image.")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt for Qwen3 chat template.")
    parser.add_argument("--user-prompt", type=str, default=DEFAULT_USER_PROMPT, help="User text that accompanies the attack image.")
    parser.add_argument("--targets-path", type=str, default="harmful_corpus/derogatory_corpus.csv", help="CSV file containing target strings.")
    return parser.parse_args()


def load_targets(path):
    with open(path, "r") as f:
        reader = csv.reader(f)
        targets = [row[0] for row in reader if row]
    if not targets:
        raise ValueError(f"No targets found in {path}")
    return targets


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    print(">>> Initializing Qwen3-VL model")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()
    print("[Initialization Finished]\n")

    targets = load_targets(args.targets_path)

    prompt_builder = PromptWrapper(
        processor=processor,
        device=device,
        system_prompt=args.system_prompt,
        user_prompt=args.user_prompt,
    )

    attacker = Attacker(
        args=args,
        model=model,
        processor=processor,
        prompt_builder=prompt_builder,
        targets=targets,
        device=device,
    )

    template = Image.open(args.template_image).convert("RGB")
    image_inputs = processor.image_processor(images=[template], return_tensors="pt")
    template_pixels = image_inputs["pixel_values"].to(device)
    template_img = attacker.denormalize(template_pixels)

    alpha = args.alpha / 255
    epsilon = args.eps / 255

    if args.constrained:
        adv_img_prompt = attacker.attack_constrained(
            img=template_img,
            batch_size=args.batch_size,
            num_iter=args.n_iters,
            alpha=alpha,
            epsilon=epsilon,
        )
    else:
        adv_img_prompt = attacker.attack_unconstrained(
            img=template_img,
            batch_size=args.batch_size,
            num_iter=args.n_iters,
            alpha=alpha,
        )

    save_path = os.path.join(args.save_dir, "bad_prompt.bmp")
    save_image(adv_img_prompt, save_path)
    print(f"[Done] Saved adversarial image to {save_path}")


if __name__ == "__main__":
    main()
