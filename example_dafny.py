#!/usr/bin/env python3
"""
Example demonstrating MCTS for Dafny code generation with verification.

This example generates Dafny code and verifies it using the Dafny verifier.
Based on the opt0 problem from llm-verified-with-monte-carlo-tree-search.
"""

import subprocess
import tempfile
import os
from libvermcts import mcts, VerifierResult


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


def dafny_verifier(base: str, add: str) -> VerifierResult:
    """
    Verifies Dafny code using the Dafny verifier.

    In MCTS, base is the prompt and add is the generated content.
    Uses the model's end token (```) to determine when generation is complete.
    Only checks entire lines (add must end with newline).
    """
    code = base + add

    # Check if we've hit the end token for Dafny code blocks
    if add.strip().endswith('```'):
        # End token found! Extract code from ``` blocks and verify
        extracted_code = extract_dafny_code(code)
        result = check_dafny(extracted_code)

        if result['status'] == 0:
            # Verification succeeded!
            return VerifierResult.DONE
        else:
            # Verification failed
            print(result)
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
        print("TOO MANY CLOSING BRACES")
        return VerifierResult.BAD
    else:
        # Braces are balanced, try verifying
        # Extract code from ``` blocks before verification
        extracted_code = extract_dafny_code(code)
        result = check_dafny(extracted_code)

        if result['status'] == 0:
            # Verification succeeded!
            return VerifierResult.GOOD
        else:
            # Verification failed with balanced braces
            # Could be incomplete or could be bad
            # If we have some content, mark as inconclusive to keep trying
            print(result)
            if len(add.strip()) < 50:
                return VerifierResult.INCONCLUSIVE
            else:
                return VerifierResult.BAD


def main():
    prompt = DAFNY_PROMPT
    model = "codellama/CodeLlama-34b-Instruct-hf"
    #model = "google/gemma-3-27b-it"

    print("=" * 70)
    print("Dafny Code Generation with MCTS and Verification")
    print("=" * 70)
    print(f"Model: {model}")
    print("-" * 70)
    print(f"Prompt:\n{prompt}")
    print("-" * 70)

    result = mcts(
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
        final_check = check_dafny(prompt + result)
        if final_check['status'] == 0:
            print("✓ Final verification: PASSED")
        else:
            print("✗ Final verification: FAILED")
            print(f"Output: {final_check['out']}")
    else:
        print("\n" + "=" * 70)
        print("Failed to generate valid Dafny code within token budget.")
        print("=" * 70)


if __name__ == "__main__":
    main()
