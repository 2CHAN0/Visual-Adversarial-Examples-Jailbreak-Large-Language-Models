import random
from typing import List, Tuple, Optional

import torch
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from qwen3_vl_utils import prompt_wrapper, generator


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device)
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images


def denormalize(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=images.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=images.device)
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images


def find_subsequence(sequence: List[int], target: List[int]) -> Optional[int]:
    if not target or len(target) > len(sequence):
        return None
    for idx in range(len(sequence) - len(target) + 1):
        if sequence[idx: idx + len(target)] == target:
            return idx
    return None


class Attacker:

    def __init__(self, args, model, tokenizer, processor, base_messages, targets, device: str = "cuda:0"):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.base_messages = base_messages
        self.targets = targets
        self.device = device

        self.generator = generator.Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            base_messages=self.base_messages,
            device=self.device
        )

        self.loss_buffer: List[float] = []

        self.model.eval()
        self.model.requires_grad_(False)

    def attack_unconstrained(self, img, batch_size=8, num_iter=2000, alpha=1 / 255.0):
        adv_noise = torch.rand_like(img).to(self.device)
        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        for t in tqdm(range(num_iter + 1)):
            batch_targets = self._sample_targets(batch_size)

            x_adv = normalize(adv_noise)
            loss = self.attack_loss(x_adv, batch_targets)
            loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(0, 1)
            adv_noise.grad.zero_()
            self.model.zero_grad(set_to_none=True)

            self._log_step(t, loss.item())
            self._maybe_log_samples(t, x_adv)

        final = normalize(adv_noise)
        adv_img_prompt = denormalize(final).detach().cpu().squeeze(0)
        return adv_img_prompt

    def attack_constrained(self, img, batch_size=8, num_iter=2000, alpha=1 / 255.0, epsilon=128 / 255.0):
        adv_noise = torch.rand_like(img).to(self.device) * 2 * epsilon - epsilon
        x = denormalize(img).clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data

        adv_noise.requires_grad_(True)
        adv_noise.retain_grad()

        for t in tqdm(range(num_iter + 1)):
            batch_targets = self._sample_targets(batch_size)
            x_adv = x + adv_noise
            x_adv = normalize(x_adv)

            loss = self.attack_loss(x_adv, batch_targets)
            loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad(set_to_none=True)

            self._log_step(t, loss.item())
            self._maybe_log_samples(t, x_adv)

        adv_img_prompt = denormalize(x_adv).detach().cpu().squeeze(0)
        return adv_img_prompt

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

        adv_img_prompt = denormalize(image).detach().cpu().squeeze(0)
        save_image(adv_img_prompt, f"{self.args.save_dir}/bad_prompt_temp_{iteration}.bmp")

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

        images = images.repeat(batch_size, 1, 1, 1)

        input_ids_list = []
        labels_list = []
        attention_list = []

        target_tokenized = self.tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )

        for idx, target in enumerate(targets):
            convo = prompt_wrapper.append_assistant_response(self.base_messages, target)
            encoded = self.tokenizer.apply_chat_template(
                convo,
                tokenize=True,
                add_generation_prompt=False,
            )

            if isinstance(encoded, dict):
                ids = encoded["input_ids"]
                attention = encoded.get("attention_mask")
            else:
                ids = encoded
                attention = None

            ids_tensor = self._ensure_tensor(ids)
            attention_tensor = self._ensure_tensor(attention) if attention is not None else torch.ones_like(ids_tensor)

            labels = ids_tensor.clone()

            target_ids = target_tokenized["input_ids"][idx]
            if self.tokenizer.pad_token_id is not None:
                target_list = target_ids[target_ids != self.tokenizer.pad_token_id].tolist()
            else:
                target_list = target_ids.tolist()
            start = find_subsequence(ids_tensor[0].tolist(), target_list)
            if start is None:
                start = max(ids_tensor.shape[1] - len(target_list), 0)

            labels[:, :start] = -100
            end = start + len(target_list)
            if end < labels.shape[1]:
                labels[:, end:] = -100

            input_ids_list.append(ids_tensor)
            labels_list.append(labels)
            attention_list.append(attention_tensor)

        input_ids, attention_mask, labels = self._pad_inputs(input_ids_list, attention_list, labels_list)

        outputs = self.model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device),
            pixel_values=images.half(),
            return_dict=True,
        )
        return outputs.loss

    def _sample_targets(self, batch_size: int) -> List[str]:
        population = len(self.targets)
        if population == 0:
            raise ValueError("Target list is empty.")
        sample_size = min(batch_size, population)
        return random.sample(self.targets, sample_size)

    @staticmethod
    def _ensure_tensor(value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, list):
            return torch.tensor(value).unsqueeze(0)
        raise ValueError("Unsupported value type for tensor conversion.")

    def _pad_inputs(
            self,
            input_ids_list: List[torch.Tensor],
            attention_list: List[torch.Tensor],
            labels_list: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_len = max(t.shape[1] for t in input_ids_list)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0

        batch = len(input_ids_list)
        input_ids = torch.full((batch, max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch, max_len), dtype=torch.long)
        labels = torch.full((batch, max_len), -100, dtype=torch.long)

        for i in range(batch):
            length = input_ids_list[i].shape[1]
            input_ids[i, :length] = input_ids_list[i][0]
            attention_mask[i, :length] = attention_list[i][0]
            labels[i, :length] = labels_list[i][0]

        return input_ids, attention_mask, labels
