#!/usr/bin/env python3
"""
Example demonstrating MCTS with logits for Dafny code generation with verification.

This example generates Dafny code and verifies it using the Dafny verifier,
while tracking token probabilities.
"""

import subprocess
import tempfile
import os
from libvermcts import mcts_logits, VerifierResult, ModelWrapper


# Prompt from problem_opt0_dafny_sanity_check in default_prompts.py
# Using chat template format for instruction-tuned model
DAFNY_PROMPT = """<start_of_turn>user
### Spec: In Dafny, write an ADT for arithmetic expressions comprising constants, variables and binary additions. Then write an evaluator taking an expression and an environment (a function that takes a variable name and returns a number) and returning the number resulting from evaluation. Then write an optimizer taking an expression and returning an expression with all additions by 0 removed. Then prove that the optimizer preserves the semantics as defined by the evaluation function.

Do not place this into a module.

<end_of_turn>
<start_of_turn>model

```dafny
"""


def check_dafny(code: str) -> dict:
    """
    Check Dafny code using the Dafny verifier.

    Returns:
        dict with 'status' (0 for success, non-zero for error) and 'out' (output)
    """
    try:
        # Write code to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.dfy', delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Run dafny verify
        result = subprocess.run(
            ['dafny', 'verify', temp_file],
            capture_output=True,
            text=True,
            timeout=10
        )

        return {
            'status': result.returncode,
            'out': result.stdout + result.stderr
        }
    except subprocess.TimeoutExpired:
        return {
            'status': -1,
            'out': 'Timeout'
        }
    except FileNotFoundError:
        return {
            'status': -1,
            'out': 'Dafny not found. Please install Dafny: https://github.com/dafny-lang/dafny'
        }
    finally:
        # Clean up temp file
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)


def extract_dafny_code(text: str) -> str:
    """
    Extract Dafny code from markdown code blocks.
    Removes ```dafny and ``` markers.
    """
    import re
    # Find all dafny code blocks
    pattern = r'```[Dd]afny\s*(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        # Join all code blocks
        return '\n'.join(matches)

    # Try to find code after ```dafny marker (even if not closed yet)
    pattern2 = r'```[Dd]afny\s*(.*?)$'
    matches2 = re.findall(pattern2, text, re.DOTALL)
    if matches2:
        return matches2[0]

    # If no code blocks found, return as-is
    return text


# Global counter for verifier calls
verifier_call_count = 0


def dafny_verifier(base: str, add: str, last_logit: float, logit_product: float) -> VerifierResult:
    """
    Verifies Dafny code using the Dafny verifier.

    In MCTS, base is the prompt and add is the generated content.
    Uses the model's end token (```) to determine when generation is complete.
    Only checks entire lines (add must end with newline).

    Args:
        base: The base prompt
        add: The generated text to verify
        last_logit: Probability of the last generated token (0.0-1.0)
        logit_product: Product of all token probabilities in add
    """
    global verifier_call_count
    verifier_call_count += 1

    code = base + add

    # Print logit information for debugging
    print(f"[Verifier call #{verifier_call_count}] last_logit={last_logit:.4f}, logit_product={logit_product:.6e}, add_len={len(add)}")

    # Check if we've hit the end token for Dafny code blocks
    if add.strip().endswith('```'):
        # End token found! Extract code from ``` blocks and verify
        extracted_code = extract_dafny_code(code)
        result = check_dafny(extracted_code)

        if result['status'] == 0:
            # Verification succeeded!
            print(f"✓ Verification PASSED! (last_logit={last_logit:.4f}, logit_product={logit_product:.6e})")
            return VerifierResult.DONE
        else:
            # Verification failed
            print(f"✗ Verification FAILED (last_logit={last_logit:.4f}, logit_product={logit_product:.6e})")
            print(f"  Error: {result['out'][:100]}...")
            return VerifierResult.BAD

    # Only check entire lines
    if not add.endswith('\n'):
        return VerifierResult.INCONCLUSIVE

    # Count opening and closing braces
    open_braces = code.count('{')
    close_braces = code.count('}')

    if close_braces < open_braces:
        # Too few closing braces - still building
        return VerifierResult.INCONCLUSIVE
    elif close_braces > open_braces:
        # Too many closing braces - bad
        print(f"✗ TOO MANY CLOSING BRACES (last_logit={last_logit:.4f}, logit_product={logit_product:.6e})")
        return VerifierResult.BAD
    else:
        # Braces are balanced, try verifying
        # Extract code from ``` blocks before verification
        extracted_code = extract_dafny_code(code)
        result = check_dafny(extracted_code)

        if result['status'] == 0:
            # Verification succeeded!
            print(f"✓ Intermediate verification PASSED (last_logit={last_logit:.4f}, logit_product={logit_product:.6e})")
            return VerifierResult.GOOD
        else:
            # Verification failed with balanced braces
            # Could be incomplete or could be bad
            # If we have some content, mark as inconclusive to keep trying
            if len(add.strip()) < 50:
                return VerifierResult.INCONCLUSIVE
            else:
                print(f"✗ Verification failed with balanced braces (last_logit={last_logit:.4f}, logit_product={logit_product:.6e})")
                print(f"  Error: {result['out'][:100]}...")
                return VerifierResult.BAD


def compute_token_logits(model_wrapper: ModelWrapper, prompt: str, generated_text: str):
    """
    Compute logits for each token in the generated text.

    Args:
        model_wrapper: The model wrapper to use for computing logits
        prompt: The base prompt
        generated_text: The generated text to analyze

    Returns:
        List of (token_text, logit) tuples
    """
    print("\nComputing logits for each token in the final result...")

    # Tokenize the generated text
    tokens = model_wrapper.tokenizer.encode(generated_text)

    token_logits = []

    # For each token, compute its probability given the previous context
    for i in range(len(tokens)):
        # Get tokens up to (but not including) current position
        context_tokens = tokens[:i]

        # Get the current token
        current_token = tokens[i]

        # Compute logit for this token
        import torch
        with torch.no_grad():
            # Prepare input
            if context_tokens:
                input_ids = torch.tensor([model_wrapper.tokenizer.encode(prompt) + context_tokens])
            else:
                input_ids = torch.tensor([model_wrapper.tokenizer.encode(prompt)])

            input_ids = input_ids.to(model_wrapper.device)

            # Get logits
            outputs = model_wrapper.model(input_ids)
            logits = outputs.logits[0, -1, :]

            # Convert to probabilities
            probs = torch.softmax(logits, dim=-1)

            # Get probability of the current token
            token_prob = probs[current_token].item()

        # Decode the token to text
        token_text = model_wrapper.tokenizer.decode([current_token])

        token_logits.append((token_text, token_prob, i))

    return token_logits


def main():
    prompt = DAFNY_PROMPT
    model = "codellama/CodeLlama-34b-Instruct-hf"
    #model = "google/gemma-3-27b-it"

    print("=" * 70)
    print("Dafny Code Generation with MCTS, Verification, and Logits")
    print("=" * 70)
    print(f"Model: {model}")
    print("-" * 70)
    print(f"Prompt:\n{prompt}")
    print("-" * 70)

    result = mcts_logits(
        prompt=prompt,
        verifier=dafny_verifier,
        model=model,
        max_tokens=2000,
        expansion_count=100,
        verbose=True
    )

    if result:
        print("\n" + "=" * 70)
        print("Success! Generated Dafny code:")
        print("=" * 70)
        print(f"{prompt}{result}")
        print("=" * 70)

        # Verify the final result
        final_code = extract_dafny_code(prompt + result)
        final_check = check_dafny(final_code)
        if final_check['status'] == 0:
            print("✓ Final verification: PASSED")
        else:
            print("✗ Final verification: FAILED")
            print(f"Output: {final_check['out']}")

        print(f"\nTotal verifier calls: {verifier_call_count}")

        # Compute logits for all tokens in the final result
        print("\n" + "=" * 70)
        print("Analyzing token probabilities in final result...")
        print("=" * 70)

        # Initialize model wrapper for analysis
        model_wrapper = ModelWrapper(model)

        # Compute logits for each token
        token_logits = compute_token_logits(model_wrapper, prompt, result)

        print(f"Total tokens in result: {len(token_logits)}")

        # Sort by logit value (ascending) to get lowest probabilities first
        sorted_tokens = sorted(token_logits, key=lambda x: x[1])

        # Display the 20 tokens with lowest logit values
        print("\n" + "=" * 70)
        print("20 Tokens with LOWEST Logit Values:")
        print("=" * 70)
        print(f"{'Rank':<6} {'Logit':<12} {'Position':<10} {'Token (repr)':<30}")
        print("-" * 70)
        for rank, (token_text, logit, position) in enumerate(sorted_tokens[:20], 1):
            token_repr = repr(token_text)[:30]  # Truncate long representations
            print(f"{rank:<6} {logit:<12.8f} {position:<10} {token_repr:<30}")

        print("=" * 70)

    else:
        print("\n" + "=" * 70)
        print("Failed to generate valid Dafny code within token budget.")
        print("=" * 70)
        print(f"\nTotal verifier calls: {verifier_call_count}")


if __name__ == "__main__":
    main()
