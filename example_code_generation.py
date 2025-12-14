#!/usr/bin/env python3
"""
Example demonstrating MCTS for code generation with syntax verification.

This example generates Python code and verifies it has valid syntax.
"""

import ast
from libvermcts import mcts, VerifierResult


def python_syntax_verifier(base: str, add: str) -> VerifierResult:
    """
    Verifies that the generated code is valid Python with a complete function.
    """
    if add.endswith("\n"):
        return VerifierResult.INCONCLUSIVE
    code = base + add

    try:
        # Try to parse as Python
        tree = ast.parse(code)

        # Check if we have at least one function definition
        has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))

        if has_function:
            # Count top-level function definitions (not nested ones)
            top_level_functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]

            if len(top_level_functions) >= 2:
                # Found a second top-level function, we're done
                return VerifierResult.DONE
            elif len(top_level_functions) == 1:
                # Have one function, still good progress
                return VerifierResult.GOOD
            else:
                # Has function but somehow not top-level (shouldn't happen)
                return VerifierResult.INCONCLUSIVE
        else:
            # Valid syntax but no function yet
            return VerifierResult.INCONCLUSIVE

    except SyntaxError as e:
        # Invalid syntax - might be incomplete or actually bad
        # If the syntax error is in the last line of the source code,
        # assume it's incomplete, otherwise assume bad
        code_lines = code.split('\n')
        total_lines = len(code_lines)

        # Check if error is on or near the last line (within last 2 lines)
        # SyntaxError.lineno gives the line number where the error occurred
        if e.lineno is not None and e.lineno >= total_lines - 1:
            # Error is at the end, likely incomplete
            return VerifierResult.INCONCLUSIVE
        else:
            # Error is in the middle, likely bad code
            return VerifierResult.BAD


def main():
    prompt = "def fibonacci(n):\n    "
    model = "google/gemma-3-1b-it"

    print(f"Prompt: {prompt}")
    print(f"Model: {model}")
    print("-" * 50)

    result = mcts(
        prompt=prompt,
        verifier=python_syntax_verifier,
        model=model,
        max_tokens=2000,
        expansion_count=100,
        verbose=True
    )

    if result:
        print(f"Success! Generated code:")
        print(f"{prompt}{result}")
        print("-" * 50)
        print("Verification: Valid Python syntax with function definition")
    else:
        print("Failed to generate valid Python code within token budget.")


if __name__ == "__main__":
    main()
