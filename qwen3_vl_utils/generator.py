import copy
from typing import List, Dict, Any

import torch


class Generator:

    def __init__(self, model, tokenizer, base_messages: List[Dict[str, Any]], device: str = "cuda:0",
                 max_new_tokens: int = 512, temperature: float = 0.2):
        self.model = model
        self.tokenizer = tokenizer
        self.base_messages = base_messages
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def _build_input_ids(self, extra_user_prompt: str = "") -> torch.Tensor:
        messages = copy.deepcopy(self.base_messages)
        if extra_user_prompt:
            messages[-1]["content"].append({"type": "text", "text": extra_user_prompt})

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.device)
        return input_ids

    def generate(self, image: torch.Tensor, extra_user_prompt: str = "") -> str:
        input_ids = self._build_input_ids(extra_user_prompt)
        input_len = input_ids.shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                pixel_values=image.half(),
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
