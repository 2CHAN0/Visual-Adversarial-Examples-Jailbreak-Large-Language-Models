import copy
from typing import List, Dict, Any

import torch

from qwen3_vl_utils import pixel_processor


class Generator:

    def __init__(self, model, tokenizer, processor, base_messages: List[Dict[str, Any]],
                 device: str = "cuda:0", max_new_tokens: int = 512,
                 temperature: float = 0.2,
                 pixel_value_builder=None):

        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.base_messages = base_messages
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        if pixel_value_builder is None:
            raise ValueError("pixel_value_builder must be provided for Generator.")
        self.pixel_value_builder = pixel_value_builder
        self.model_dtype = next(self.model.parameters()).dtype

    def _build_messages(self, extra_user_prompt: str = "") -> List[Dict[str, Any]]:
        messages = copy.deepcopy(self.base_messages)
        if extra_user_prompt:
            messages[-1]["content"].append({"type": "text", "text": extra_user_prompt})
        return messages

    def generate(self, image: torch.Tensor, extra_user_prompt: str = "") -> str:
        image_input = self._ensure_image_batch(image)

        messages = self._build_messages(extra_user_prompt)
        prompt_str = self._build_prompt_string(messages)
        pil_images = pixel_processor.pixel_tensor_to_pil(image_input)

        pixel_values, image_grid_thw = self.pixel_value_builder.build_inputs(image_input.to(self.device))
        pixel_values = pixel_values.to(device=self.device, dtype=self.model_dtype)
        image_grid_thw = image_grid_thw.to(self.device)

        processor_inputs = self.processor(
            images=pil_images,
            text=[prompt_str],
            return_tensors="pt",
            padding=True,
        )

        processor_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in processor_inputs.items()
        }
        processor_inputs["pixel_values"] = pixel_values
        processor_inputs["image_grid_thw"] = image_grid_thw

        input_ids = processor_inputs["input_ids"]
        input_len = input_ids.shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **processor_inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True
        )[0]
        return outputs.strip()

    def _ensure_image_batch(self, tensor: torch.Tensor) -> torch.Tensor:
        image_input = tensor.detach()
        if image_input.dim() >= 5:
            image_input = image_input[0:1]
        if image_input.dim() == 3:
            image_input = image_input.unsqueeze(0)
        if image_input.dim() != 4:
            raise ValueError(f"Unsupported image tensor shape {image_input.shape} for generator.")
        return image_input.clamp(0.0, 1.0)

    def _build_prompt_string(self, messages: List[Dict[str, Any]]) -> str:
        num_images = self._count_images(messages)
        image_placeholders = [None] * num_images if num_images > 0 else None
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            images=image_placeholders,
        )
        return prompt

    @staticmethod
    def _count_images(messages: List[Dict[str, Any]]) -> int:
        count = 0
        for turn in messages:
            content = turn.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    count += 1
        return count
