import torch

DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless visual assistant."
DEFAULT_USER_PROMPT = ""


class PromptWrapper:
    """Builds tokenized prompts for Qwen3-VL chat templates."""

    def __init__(self, processor, device, system_prompt=DEFAULT_SYSTEM_PROMPT, user_prompt=DEFAULT_USER_PROMPT):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.device = device
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt

        # Ensure padding happens on the right so label masking aligns with chat template.
        self.tokenizer.padding_side = "right"

        self.context_length = self._compute_context_length()

    def _messages(self, assistant_text=None):
        """Constructs the message list expected by the chat template."""
        user_content = [{"type": "image"}]
        if self.user_prompt:
            user_content.append({"type": "text", "text": self.user_prompt})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        if assistant_text is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

        return messages

    def _compute_context_length(self):
        """Token count of the conversation prefix before assistant text begins."""
        prefix = self.processor.apply_chat_template(
            self._messages(), tokenize=False, add_generation_prompt=True
        )
        tokenized = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
        return tokenized.input_ids.shape[-1]

    def build_batch_inputs(self, targets):
        """Tokenizes assistant responses and returns tensors suitable for supervised loss."""
        conversations = [
            self.processor.apply_chat_template(
                self._messages(target), tokenize=False, add_generation_prompt=False
            )
            for target in targets
        ]

        tokenized = self.tokenizer(conversations, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        labels = input_ids.clone()

        pad_id = self.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        context_tokens = min(self.context_length, labels.shape[1])
        labels[:, :context_tokens] = -100

        return input_ids, attention_mask, labels

    def build_generation_inputs(self):
        """Returns tokenized inputs for greedy/sampled generation."""
        prompt = self.processor.apply_chat_template(
            self._messages(), tokenize=False, add_generation_prompt=True
        )
        tokenized = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        return tokenized
