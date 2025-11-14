import argparse
import csv
import os

from PIL import Image
import torch
from torchvision.utils import save_image

from qwen3_vl_utils import prompt_wrapper, visual_attacker
from qwen3_vl_utils.model_loader import load_qwen_model


def parse_args():

    parser = argparse.ArgumentParser(description="Visual attack script for Qwen3-VL models.")
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-VL-2B-Instruct",
                        help="Hugging Face identifier of the Qwen3-VL checkpoint.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU id used for model loading.")
    parser.add_argument("--n-iters", type=int, default=5000, help="Number of optimization steps.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of target strings per step.")
    parser.add_argument("--eps", type=int, default=32, help="L-infinity budget (0~255 scale) for constrained attack.")
    parser.add_argument("--alpha", type=int, default=1, help="Per-step update size (0~255 scale).")
    parser.add_argument("--constrained", default=False, action="store_true",
                        help="Enable constrained attack mode.")

    parser.add_argument("--system-prompt", type=str, default="You are a helpful assistant.",
                        help="System prompt prepended to every conversation.")
    parser.add_argument("--user-prompt", type=str, default="",
                        help="Optional user text that accompanies the adversarial image.")
    parser.add_argument("--template-image", type=str, default="adversarial_images/clean.jpeg",
                        help="Starting image for optimization.")

    parser.add_argument("--targets-file", type=str, default="harmful_corpus/derogatory_corpus.csv",
                        help="CSV file containing target continuations (single column).")
    parser.add_argument("--save-dir", type=str, default="output",
                        help="Destination directory for attack logs and the final image.")

    args = parser.parse_args()
    return args


def load_image(path):
    return Image.open(path).convert("RGB")


def read_targets(csv_path):
    with open(csv_path, "r") as fh:
        reader = csv.reader(fh, delimiter=",")
        targets = [row[0] for row in reader if row]
    if not targets:
        raise ValueError(f"No targets found in {csv_path}")
    return targets


def main():

    print(">>> Initializing Qwen3-VL pipeline")
    args = parse_args()

    tokenizer, processor, model, device = load_qwen_model(args.model_name, args.gpu_id)
    base_messages = prompt_wrapper.build_base_messages(args.system_prompt, args.user_prompt)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    template = load_image(args.template_image)
    image_tensor = processor.image_processor(template, return_tensors="pt")["pixel_values"].to(device=device,
                                                                                                 dtype=model.dtype)

    targets = read_targets(args.targets_file)
    attacker = visual_attacker.Attacker(
        args=args,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        base_messages=base_messages,
        targets=targets,
        device=device,
    )

    if not args.constrained:
        print("[Qwen3-VL][unconstrained attack]")
        adv = attacker.attack_unconstrained(
            img=image_tensor,
            batch_size=args.batch_size,
            num_iter=args.n_iters,
            alpha=args.alpha / 255.0,
        )
    else:
        print("[Qwen3-VL][constrained attack]")
        adv = attacker.attack_constrained(
            img=image_tensor,
            batch_size=args.batch_size,
            num_iter=args.n_iters,
            alpha=args.alpha / 255.0,
            epsilon=args.eps / 255.0,
        )

    save_path = os.path.join(args.save_dir, "bad_prompt.bmp")
    save_image(adv, save_path)
    print(f"[Done] Saved adversarial image to {save_path}")


if __name__ == "__main__":
    main()
