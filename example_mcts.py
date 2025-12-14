#!/usr/bin/env python3
"""
Example demonstrating MCTS-based verified text generation.

This example uses MCTS to generate text with incremental verification,
allowing the algorithm to backtrack when it generates bad content.
"""

from libvermcts import mcts, VerifierResult

def simple_sentence_verifier(base: str, add: str) -> VerifierResult:
    """
    A simple verifier that checks for a complete sentence with reasonable length.
    In MCTS, base is the prompt and add is the generated content.
    """
    # add already contains just the generated part
    generated = add.strip()

    # Check if we have too many characters without actual words (garbage detection)
    if len(generated) > 100 and len(generated.split()) < 3:
        return VerifierResult.BAD

    # Too short
    if len(generated.split()) < 3:
        return VerifierResult.INCONCLUSIVE

    # Check for sentence ending
    if generated.endswith('.'):
        # Has proper ending
        word_count = len(generated.split())

        if 5 <= word_count <= 20:
            return VerifierResult.DONE
        elif word_count > 20:
            return VerifierResult.BAD
        else:
            return VerifierResult.INCONCLUSIVE

    # Building up
    word_count = len(generated.split())
    if word_count > 25:
        # Too long without ending
        return VerifierResult.BAD

    return VerifierResult.INCONCLUSIVE


def main():
    print("=" * 60)
    print("MCTS Example 1: Simple Sentence Generation")
    print("=" * 60)

    prompt1 = "Write a sentence about cats: "
    model = "google/gemma-3-1b-it"

    print(f"Prompt: {prompt1}")
    print(f"Model: {model}")
    print("-" * 60)

    result1 = mcts(
        prompt=prompt1,
        verifier=simple_sentence_verifier,
        model=model,
        max_tokens=500,
        expansion_count=50,
        verbose=True
    )

    if result1:
        print(f"Success! Generated text:")
        print(f"{result1}")
    else:
        print("Failed to generate valid text within token budget.")

    print("\n")
    print("=" * 60)


if __name__ == "__main__":
    main()
