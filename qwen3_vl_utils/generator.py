import copy
from typing import List, Dict, Any, Optional

import torch


class Generator:

    def __init__(self, model, processor, base_messages: List[Dict[str, Any]], device: str = "cuda:0",
                 max_new_tokens: int = 512, temperature: float = 0.2):
        self.model = model
        self.processor = processor
        self.base_messages = base_messages
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _build_inputs(self, image: torch.Tensor, extra_user_prompt: str = ""):
        messages = copy.deepcopy(self.base_messages)
        if extra_user_prompt:
            messages[-1]["content"].append({"type": "text", "text": extra_user_prompt})

        image = image.detach()
        if image.dim() >= 5:
            image = image[0]
        if image.dim() == 3:
            image_input = image
        elif image.dim() == 4:
            image_input = image.squeeze(0)
        else:
            raise ValueError(f"Unsupported image tensor shape {image.shape} for generator.")

        batch = self.processor(
            images=[image_input.cpu()],
            text=messages,
            return_tensors="pt"
        )
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        return batch

    def generate(self, image: torch.Tensor, extra_user_prompt: str = "") -> str:
        inputs = self._build_inputs(image, extra_user_prompt)
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )

        outputs = self.processor.tokenizer.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True
        )[0]
        return outputs.strip()
