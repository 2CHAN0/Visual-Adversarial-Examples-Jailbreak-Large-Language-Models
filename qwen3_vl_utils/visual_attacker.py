import random

import torch
from torchvision.utils import save_image
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

from qwen3_vl_utils import generator


class Attacker:
    """Projected Gradient Descent style attack on Qwen3-VL models."""

    def __init__(self, args, model, processor, prompt_builder, targets, device="cuda:0"):
        self.args = args
        self.model = model
        self.processor = processor
        self.prompt_builder = prompt_builder
        self.targets = targets
        self.device = device
        self.loss_buffer = []

        self.model.eval()
        self.model.requires_grad_(False)

        self.generator = generator.Generator(model, processor, prompt_builder, device=device)

        image_proc = processor.image_processor
        mean = torch.tensor(image_proc.image_mean, dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(image_proc.image_std, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device))

        self.model_dtype = next(model.parameters()).dtype

    def register_buffer(self, name, tensor):
        setattr(self, name, tensor)

    def normalize(self, images):
        return (images - self.mean) / self.std

    def denormalize(self, images):
        return images * self.std + self.mean

    def attack_unconstrained(self, img, batch_size=8, num_iter=2000, alpha=1 / 255):
        adv_noise = torch.rand_like(img).to(self.device)
        adv_noise.requires_grad_(True)

        final_adv = None

        for t in tqdm(range(num_iter + 1)):
            batch_targets = self.sample_targets(batch_size)
            pixel_values = self.normalize(adv_noise)
            pixel_batch = pixel_values.repeat(batch_size, 1, 1, 1).to(self.device, dtype=self.model_dtype)

            target_loss = self.attack_loss(pixel_batch, batch_targets)
            target_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(0, 1)
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())
            print(f"target_loss: {target_loss.item():.6f}")

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                final_adv = self.log_progress(t, adv_noise)

        if final_adv is None:
            final_adv = (
                self.denormalize(self.normalize(adv_noise)).detach().cpu().squeeze(0).clamp(0, 1)
            )

        return final_adv

    def attack_constrained(self, img, batch_size=8, num_iter=2000, alpha=1 / 255, epsilon=128 / 255):
        adv_noise = (torch.rand_like(img) * 2 * epsilon - epsilon).to(self.device)
        x = img.clone().to(self.device)
        adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
        adv_noise.requires_grad_(True)

        final_adv = None

        for t in tqdm(range(num_iter + 1)):
            batch_targets = self.sample_targets(batch_size)

            x_adv = x + adv_noise
            pixel_values = self.normalize(x_adv)
            pixel_batch = pixel_values.repeat(batch_size, 1, 1, 1).to(self.device, dtype=self.model_dtype)

            target_loss = self.attack_loss(pixel_batch, batch_targets)
            target_loss.backward()

            adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
            adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
            adv_noise.grad.zero_()
            self.model.zero_grad()

            self.loss_buffer.append(target_loss.item())
            print(f"target_loss: {target_loss.item():.6f}")

            if t % 20 == 0:
                self.plot_loss()

            if t % 100 == 0:
                final_adv = self.log_progress(t, x + adv_noise)

        if final_adv is None:
            final_adv = (
                self.denormalize(self.normalize(x + adv_noise)).detach().cpu().squeeze(0).clamp(0, 1)
            )

        return final_adv

    def log_progress(self, iteration, current_img):
        pixel_values = self.normalize(current_img).detach()
        with torch.no_grad():
            response = self.generator.generate(pixel_values)
        print(f"######### Output - Iter = {iteration} ##########")
        print(">>>", response)

        adv_img_prompt = self.denormalize(pixel_values).detach().cpu().squeeze(0).clamp(0, 1)
        save_image(adv_img_prompt, f"{self.args.save_dir}/bad_prompt_temp_{iteration}.bmp")
        return adv_img_prompt

    def plot_loss(self):
        sns.set_theme()
        num_iters = len(self.loss_buffer)
        x_ticks = list(range(num_iters))
        plt.plot(x_ticks, self.loss_buffer, label="Target Loss")
        plt.title("Loss Plot")
        plt.xlabel("Iters")
        plt.ylabel("Loss")
        plt.legend(loc="best")
        plt.savefig(f"{self.args.save_dir}/loss_curve.png")
        plt.clf()
        torch.save(self.loss_buffer, f"{self.args.save_dir}/loss")

    def attack_loss(self, pixel_values, targets):
        input_ids, attention_mask, labels = self.prompt_builder.build_batch_inputs(targets)
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }
        outputs = self.model(**batch)
        return outputs.loss

    def sample_targets(self, batch_size):
        if len(self.targets) >= batch_size:
            return random.sample(self.targets, batch_size)
        return random.choices(self.targets, k=batch_size)
