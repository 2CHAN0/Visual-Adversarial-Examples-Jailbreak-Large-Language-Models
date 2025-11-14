import copy
from typing import List, Dict, Any, Optional

import torch


class Generator:

    def __init__(self, model, tokenizer, base_messages: List[Dict[str, Any]],
                 device: str = "cuda:0", max_new_tokens: int = 512,
                 temperature: float = 0.2,
                 pixel_mask: Optional[torch.Tensor] = None,
                 image_grid_thw: Optional[torch.Tensor] = None):

        self.model = model
        self.tokenizer = tokenizer
        self.base_messages = base_messages
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.pixel_mask = pixel_mask
        self.image_grid_thw = image_grid_thw

    def _build_prompt(self, extra_user_prompt: str = "") -> str:
        messages = copy.deepcopy(self.base_messages)
        if extra_user_prompt:
            messages[-1]["content"].append({"type": "text", "text": extra_user_prompt})
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return prompt

    def _build_input_ids(self, extra_user_prompt: str = "") -> torch.Tensor:
        prompt = self._build_prompt(extra_user_prompt)
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).input_ids.to(self.device)
        return input_ids

    def generate(self, image: torch.Tensor, extra_user_prompt: str = "") -> str:
        image_input = image.detach()
        if image_input.dim() >= 5:
            image_input = image_input[0:1]
        elif image_input.dim() == 4:
            image_input = image_input[:1]
        else:
            raise ValueError(f"Unsupported image tensor shape {image_input.shape} for generator.")

        input_ids = self._build_input_ids(extra_user_prompt)
        input_len = input_ids.shape[1]

        model_kwargs = {
            "pixel_values": image_input.half(),
        }
        if self.pixel_mask is not None:
            model_kwargs["pixel_mask"] = self.pixel_mask
        if self.image_grid_thw is not None:
            model_kwargs["image_grid_thw"] = self.image_grid_thw

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                **model_kwargs,
            )

        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_len:],
            skip_special_tokens=True
        )[0]
        return outputs.strip()
