import copy
from typing import List, Dict, Any, Optional

import torch


class Generator:

    def __init__(self, model, tokenizer, processor, base_messages: List[Dict[str, Any]],
                 device: str = "cuda:0", max_new_tokens: int = 512,
                 temperature: float = 0.2,
                 pixel_mask: Optional[torch.Tensor] = None,
                 image_grid_thw: Optional[torch.Tensor] = None):

        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.base_messages = base_messages
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.pixel_mask = pixel_mask
        self.image_grid_thw = image_grid_thw

    def _build_messages(self, extra_user_prompt: str = "") -> List[Dict[str, Any]]:
        messages = copy.deepcopy(self.base_messages)
        if extra_user_prompt:
            messages[-1]["content"].append({"type": "text", "text": extra_user_prompt})
        return messages

    def generate(self, image: torch.Tensor, extra_user_prompt: str = "") -> str:
        image_input = image.detach()
        if image_input.dim() >= 5:
            image_input = image_input[0:1]
        elif image_input.dim() == 4:
            image_input = image_input[:1]
        else:
            raise ValueError(f"Unsupported image tensor shape {image_input.shape} for generator.")

        messages = self._build_messages(extra_user_prompt)
        pil_images = self._prepare_visual_batch(image_input)

        processor_inputs = self.processor(
            images=pil_images,
            text=[messages],
            return_tensors="pt",
            padding=True,
        )

        processor_inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in processor_inputs.items()
        }
        processor_inputs["pixel_values"] = image_input.half()
        if self.pixel_mask is not None:
            processor_inputs["pixel_mask"] = self.pixel_mask
        if self.image_grid_thw is not None:
            processor_inputs["image_grid_thw"] = self.image_grid_thw

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

    def _prepare_visual_batch(self, tensor: torch.Tensor):
        out = tensor.detach().float().cpu()
        if out.dim() >= 5:
            out = out[:, 0]
        if out.dim() == 3:
            out = out.unsqueeze(0)
        if out.dim() != 4:
            raise ValueError(f"Unsupported tensor shape for visual batch: {tensor.shape}")
        return self.processor.image_processor.postprocess(out, output_type="pil")
