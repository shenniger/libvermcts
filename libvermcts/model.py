"""
Model wrapper for PyTorch-based text generation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelWrapper:
    """
    Wrapper for PyTorch language models that provides a simple generate interface.
    """

    def __init__(self, model_name: str):
        """
        Initialize the model wrapper.

        Args:
            model_name: HuggingFace model name (e.g., "gpt2", "facebook/opt-125m")
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Set pad_token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 1., top_p: float = 0.95) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum number of new tokens to generate (default: 512)
            temperature: Sampling temperature (default: 0.8)
            top_p: Nucleus sampling parameter (default: 0.95)

        Returns:
            Generated text (excluding the prompt)
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        # Decode only the new tokens (skip the prompt tokens)
        new_token_ids = outputs[0][prompt_length:]
        generated_text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        return generated_text

    def generate_one_token(self, prompt: str, temperature: float = 0.8, top_p: float = 0.95) -> str:
        """
        Generate a single token from a prompt.

        Args:
            prompt: Input prompt string
            temperature: Sampling temperature (default: 0.8)
            top_p: Nucleus sampling parameter (default: 0.95)

        Returns:
            The next token as a string
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        prompt_length = inputs["input_ids"].shape[1]

        # Generate one token
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
            )

        # Decode only the new token (skip the prompt tokens)
        new_token_ids = outputs[0][prompt_length:]
        new_token = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)

        return new_token
