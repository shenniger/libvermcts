"""
Whole sampling implementation.
"""

from .verifier import VerifierResult
from .model import ModelWrapper


def wholesampling(prompt: str, verifier, model: str, max_tokens: int = 10000, verbose: bool = False):
    """
    Whole sampling: Generate text and verify it using a verifier function.

    Args:
        prompt: Input prompt string
        verifier: Verifier function that takes (base, add) and returns VerifierResult
        model: Model name string (e.g., "gpt2", "facebook/opt-125m")
        max_tokens: Maximum total number of tokens to generate across all attempts (default: 10000)
        verbose: If True, print all generation attempts and verifier states (default: False)

    Returns:
        The verified generated text, or None if verification failed after max_tokens

    The function generates text from the model and calls verifier("", generation).
    If the verifier does not return DONE, it tries again in a loop until max_tokens is exhausted.
    """

    # ANSI color codes
    CYAN = '\033[96m'
    RESET = '\033[0m'

    model_wrapper = ModelWrapper(model)
    tokens_used = 0
    attempt = 0

    while tokens_used < max_tokens:
        attempt += 1
        generation = model_wrapper.generate(prompt, max_new_tokens=max_tokens - tokens_used)
        tokens_in_generation = len(model_wrapper.tokenizer.encode(generation))
        tokens_used += tokens_in_generation
        result = verifier("", generation)

        if verbose:
            print(f"\n{'='*60}")
            print(f"Attempt {attempt} (tokens used: {tokens_used}/{max_tokens})")
            print(f"{'='*60}")
            print(f"Generated text:\n{prompt}{CYAN}{generation}{RESET}")
            print(f"{'-'*60}")
            print(f"Verifier result: {result.value}")
            print(f"{'='*60}")

        if result == VerifierResult.DONE:
            return generation

    return None
