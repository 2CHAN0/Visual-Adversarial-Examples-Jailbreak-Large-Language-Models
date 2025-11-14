import torch


class Generator:
    """Utility to log sample generations during optimization."""

    def __init__(
        self,
        model,
        processor,
        prompt_wrapper,
        device,
        max_new_tokens=256,
        top_p=0.9,
        temperature=1.0,
    ):
        self.model = model
        self.processor = processor
        self.prompt_wrapper = prompt_wrapper
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.tokenizer = processor.tokenizer

    def generate(self, pixel_values):
        inputs = self.prompt_wrapper.build_generation_inputs()
        inputs["pixel_values"] = pixel_values.to(self.device, dtype=self.model.dtype)

        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        context_length = inputs["input_ids"].shape[-1]
        generated_tokens = output_ids[:, context_length:]
        text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        return text
