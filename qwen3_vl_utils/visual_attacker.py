import os
import random
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from qwen3_vl_utils import prompt_wrapper, generator, pixel_processor


def find_subsequence(sequence: List[int], target: List[int]) -> Optional[int]:
    if not target or len(target) > len(sequence):
        return None
    for idx in range(len(sequence) - len(target) + 1):
        if sequence[idx: idx + len(target)] == target:
            return idx
    return None


class Attacker:

    def __init__(self, args, model, tokenizer, processor, base_messages, targets,
                 device: str = "cuda:0"):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.base_messages = base_messages
        self.targets = targets
        self.device = device
        self.torch_device = torch.device(device)

        self.pixel_value_builder = pixel_processor.PixelValueBuilder(
            image_processor=self.processor.image_processor,
            device=self.torch_device,
        )

        vision_cfg = getattr(self.model.config, "vision_config", None)
        self.patch_size = getattr(vision_cfg, "patch_size", 14) if vision_cfg is not None else 14

        self.generator = generator.Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            processor=self.processor,
            base_messages=self.base_messages,
            device=self.device,
            pixel_value_builder=self.pixel_value_builder,
        )

        self.loss_buffer: List[float] = []

        self.model.eval()
        self.model.requires_grad_(False)

    def attack_unconstrained(self, img, batch_size=8, num_iter=2000, alpha=1 / 255.0):
        img = img.to(self.torch_device).clamp(0.0, 1.0)
        data_min, data_max = self._compute_bounds(img)
        adv_data = torch.empty_like(img).uniform_(data_min, data_max)
        adv_data.requires_grad_(True)

        for t in tqdm(range(num_iter + 1)):
            batch_targets = self._sample_targets(batch_size)

            loss = self.attack_loss(adv_data, batch_targets)
            loss.backward()

            adv_data.data = (adv_data.data - alpha * adv_data.grad.detach().sign()).clamp_(data_min, data_max)
            adv_data.grad.zero_()
            self.model.zero_grad(set_to_none=True)

            self._log_step(t, loss.item())
            self._maybe_log_samples(t, adv_data)

        return adv_data.detach()

    def attack_constrained(self, img, batch_size=8, num_iter=2000, alpha=1 / 255.0, epsilon=128 / 255.0):
        img = img.to(self.torch_device).clamp(0.0, 1.0)
        base_img = img.detach()
        data_min, data_max = self._compute_bounds(base_img)

        adv_data = base_img + torch.empty_like(base_img).uniform_(-epsilon, epsilon)
        adv_data = adv_data.clamp(data_min, data_max)
        adv_data.requires_grad_(True)

        for t in tqdm(range(num_iter + 1)):
            batch_targets = self._sample_targets(batch_size)

            loss = self.attack_loss(adv_data, batch_targets)
            loss.backward()

            adv_data.data = adv_data.data - alpha * adv_data.grad.detach().sign()
            adv_data.data = torch.max(torch.min(adv_data.data, base_img + epsilon), base_img - epsilon)
            adv_data.data = adv_data.data.clamp_(data_min, data_max)

            adv_data.grad.zero_()
            self.model.zero_grad(set_to_none=True)

            self._log_step(t, loss.item())
            self._maybe_log_samples(t, adv_data)

        return adv_data.detach()

    def _log_step(self, iteration: int, loss_value: float):
        self.loss_buffer.append(loss_value)
        print(f"target_loss: {loss_value:.6f}")
        if iteration % 20 == 0:
            self.plot_loss()

    def _maybe_log_samples(self, iteration: int, image: torch.Tensor):
        if iteration % 100 != 0:
            return

        try:
            response = self.generator.generate(image)
            print(f"######### Output - Iter = {iteration} ##########")
            print(">>>", response)
        except Exception as exc:
            print(f"[Warning] Failed to decode response at iter {iteration}: {exc}")

        out_path = os.path.join(self.args.save_dir, f"bad_prompt_temp_{iteration}.bmp")
        self.export_image(image, out_path)

    def plot_loss(self):
        sns.set_theme()
        x_ticks = list(range(len(self.loss_buffer)))
        plt.plot(x_ticks, self.loss_buffer, label="Target Loss")
        plt.title("Loss Plot")
        plt.xlabel("Iters")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.savefig(f"{self.args.save_dir}/loss_curve.png")
        plt.clf()
        torch.save(self.loss_buffer, f"{self.args.save_dir}/loss")

    def attack_loss(self, images: torch.Tensor, targets: List[str]) -> torch.Tensor:
        batch_size = len(targets)
        if batch_size == 0:
            raise ValueError("No targets provided for attack_loss.")

        repeat_shape = [batch_size] + [1] * (images.dim() - 1)
        pixel_batch = images.repeat(*repeat_shape).clamp(0.0, 1.0)

        pixel_values, image_grid_thw = self.pixel_value_builder.build_inputs(pixel_batch)
        pixel_values = pixel_values.to(device=self.torch_device, dtype=self.model.dtype)
        image_grid_thw = image_grid_thw.to(self.torch_device)

        batch_messages = [prompt_wrapper.append_assistant_response(self.base_messages, tgt) for tgt in targets]

        text_prompts = []
        for msg in batch_messages:
            num_images = self._count_images(msg)
            image_placeholders = [None] * num_images if num_images > 0 else None
            prompt = self.tokenizer.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=False,
                images=image_placeholders,
            )
            text_prompts.append(prompt)

        pil_images = pixel_processor.pixel_tensor_to_pil(pixel_batch)

        processor_inputs = self.processor(
            text=text_prompts,
            images=pil_images,
            return_tensors="pt",
            padding=True,
        )

        processor_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in processor_inputs.items()
        }

        input_ids = processor_inputs["input_ids"]
        attention_mask = processor_inputs["attention_mask"]
        labels = input_ids.clone()

        pixel_mask = processor_inputs.get("pixel_mask")
        if pixel_mask is None:
            pixel_mask = torch.ones(batch_size, 1, dtype=torch.long, device=self.torch_device)
        else:
            pixel_mask = pixel_mask.to(self.torch_device)

        processor_inputs["pixel_values"] = pixel_values
        processor_inputs["image_grid_thw"] = image_grid_thw

        target_tokenized = self.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        for idx in range(batch_size):
            target_ids = target_tokenized["input_ids"][idx]
            if self.tokenizer.pad_token_id is not None:
                target_tokens = target_ids[target_ids != self.tokenizer.pad_token_id].tolist()
            else:
                target_tokens = target_ids.tolist()
            target_len = len(target_tokens)
            if target_len == 0:
                labels[idx] = -100
                continue
            input_list = input_ids[idx].tolist()
            start = find_subsequence(input_list, target_tokens)
            if start is None:
                start = max(input_ids[idx].shape[0] - target_len, 0)
            if start < 0:
                start = 0
            end = min(start + target_len, input_ids[idx].shape[0])
            labels[idx, :start] = -100
            labels[idx, end:] = -100

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            image_grid_thw=image_grid_thw,
            labels=labels,
        )

        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss

        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        vocab_size = logits.shape[-1]
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        return loss_fct(logits.view(-1, vocab_size), labels.view(-1))

    def _sample_targets(self, batch_size: int) -> List[str]:
        population = len(self.targets)
        if population == 0:
            raise ValueError("Target list is empty.")
        sample_size = min(batch_size, population)
        return random.sample(self.targets, sample_size)

    def _compute_bounds(self, tensor: torch.Tensor) -> Tuple[float, float]:
        return 0.0, 1.0

    def export_image(self, tensor: torch.Tensor, path: str):
        try:
            batch = self._ensure_image_batch(tensor)
            images = pixel_processor.pixel_tensor_to_pil(batch)
            images[0].save(path)
        except Exception as exc:
            torch.save(tensor.detach().cpu(), path + ".pt")
            print(f"[Warning] Unable to render image ({exc}); raw tensor dumped to {path}.pt")

    def _ensure_image_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        batch = tensor.detach()
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        if batch.dim() != 4:
            raise ValueError(f"Unsupported tensor shape for visual batch: {tensor.shape}")
        return batch

    def _count_images(self, messages: List[Dict[str, Any]]) -> int:
        count = 0
        for turn in messages:
            content = turn.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    count += 1
        return count
