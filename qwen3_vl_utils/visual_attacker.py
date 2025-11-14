import os
import random
from typing import List, Tuple, Optional, Dict, Any

import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

from qwen3_vl_utils import prompt_wrapper, generator


def find_subsequence(sequence: List[int], target: List[int]) -> Optional[int]:
    if not target or len(target) > len(sequence):
        return None
    for idx in range(len(sequence) - len(target) + 1):
        if sequence[idx: idx + len(target)] == target:
            return idx
    return None


class Attacker:

    def __init__(self, args, model, tokenizer, processor, base_messages, targets,
                 image_meta: Optional[Dict[str, Any]] = None,
                 device: str = "cuda:0"):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.base_messages = base_messages
        self.targets = targets
        self.device = device

        image_meta = image_meta or {}
        self.pixel_mask_base = self._prepare_meta_tensor(image_meta.get("pixel_mask"))
        self.image_grid_thw_base = self._prepare_meta_tensor(image_meta.get("image_grid_thw"))

        vision_cfg = getattr(self.model.config, "vision_config", None)
        self.patch_size = getattr(vision_cfg, "patch_size", 14) if vision_cfg is not None else 14

        self.generator = generator.Generator(
            model=self.model,
            tokenizer=self.tokenizer,
            base_messages=self.base_messages,
            device=self.device,
            pixel_mask=self.pixel_mask_base,
            image_grid_thw=self.image_grid_thw_base,
        )

        self.loss_buffer: List[float] = []

        self.model.eval()
        self.model.requires_grad_(False)

    def attack_unconstrained(self, img, batch_size=8, num_iter=2000, alpha=1 / 255.0):
        data_min, data_max = self._compute_bounds(img)
        adv_data = torch.empty_like(img).uniform_(data_min, data_max).to(self.device)
        adv_data.requires_grad_(True)

        for t in tqdm(range(num_iter + 1)):
            batch_targets = self._sample_targets(batch_size)

            loss = self.attack_loss(adv_data, batch_targets)
            loss.backward()

            adv_data.data = (adv_data.data - alpha * adv_data.grad.detach().sign()).clamp(data_min, data_max)
            adv_data.grad.zero_()
            self.model.zero_grad(set_to_none=True)

            self._log_step(t, loss.item())
            self._maybe_log_samples(t, adv_data)

        return adv_data.detach()

    def attack_constrained(self, img, batch_size=8, num_iter=2000, alpha=1 / 255.0, epsilon=128 / 255.0):
        base_img = img.detach()
        data_min, data_max = self._compute_bounds(base_img)

        adv_data = base_img + torch.empty_like(base_img).uniform_(-epsilon, epsilon)
        adv_data = adv_data.clamp(data_min, data_max).to(self.device)
        adv_data.requires_grad_(True)

        for t in tqdm(range(num_iter + 1)):
            batch_targets = self._sample_targets(batch_size)

            loss = self.attack_loss(adv_data, batch_targets)
            loss.backward()

            adv_data.data = adv_data.data - alpha * adv_data.grad.detach().sign()
            adv_data.data = torch.max(torch.min(adv_data.data, base_img + epsilon), base_img - epsilon)
            adv_data.data = adv_data.data.clamp(data_min, data_max)

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
        images = images.repeat(*repeat_shape)

        pixel_mask = self._repeat_tensor(self.pixel_mask_base, batch_size)
        if pixel_mask is None:
            pixel_mask = self._default_pixel_mask(images, batch_size)

        image_grid_thw = self._repeat_tensor(self.image_grid_thw_base, batch_size)
        if image_grid_thw is None:
            image_grid_thw = self._infer_grid_thw(images, batch_size)

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
            pixel_mask=pixel_mask,
            image_grid_thw=image_grid_thw,
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

    def _compute_bounds(self, tensor: torch.Tensor) -> Tuple[float, float]:
        return tensor.min().item(), tensor.max().item()

    def _prepare_meta_tensor(self, tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            return tensor.to(self.device)
        return torch.tensor(tensor, device=self.device)

    def _repeat_tensor(self, tensor: Optional[torch.Tensor], repeat: int) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        repeat_dims = [repeat] + [1] * (tensor.dim() - 1)
        return tensor.repeat(*repeat_dims)

    def _default_pixel_mask(self, images: torch.Tensor, batch_size: int) -> torch.Tensor:
        if images.dim() >= 5:
            num_views = images.shape[1]
        else:
            num_views = 1
        return torch.ones(batch_size, num_views, dtype=torch.long, device=images.device)

    def _infer_grid_thw(self, images: torch.Tensor, batch_size: int) -> torch.Tensor:
        if images.dim() >= 5:
            _, num_views, _, h, w = images.shape
            grid = torch.zeros(batch_size, num_views, 3, dtype=torch.long, device=images.device)
            grid[:, :, 0] = 1
            grid[:, :, 1] = h // self.patch_size
            grid[:, :, 2] = w // self.patch_size
            return grid
        elif images.dim() == 4:
            _, _, h, w = images.shape
            grid = torch.tensor([1, h // self.patch_size, w // self.patch_size],
                                dtype=torch.long, device=images.device)
            return grid.unsqueeze(0).repeat(batch_size, 1)
        else:
            raise ValueError(f"Cannot infer grid from tensor with shape {images.shape}")

    def _prepare_visual_tensor(self, tensor: torch.Tensor) -> Optional[torch.Tensor]:
        out = tensor.detach().float().cpu()
        if out.dim() >= 5:
            out = out[:, 0]
        if out.dim() == 3:
            out = out.unsqueeze(0)
        if out.dim() != 4:
            return None
        return out

    def export_image(self, tensor: torch.Tensor, path: str):
        prepared = self._prepare_visual_tensor(tensor)
        if prepared is None:
            torch.save(tensor.detach().cpu(), path + ".pt")
            print(f"[Warning] Unable to render image; raw tensor dumped to {path}.pt")
            return
        images = self.processor.image_processor.postprocess(prepared, output_type="pil")
        images[0].save(path)
