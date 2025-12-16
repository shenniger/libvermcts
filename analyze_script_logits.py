#!/usr/bin/env python3
"""
Script to analyze logits/probabilities for tokens in a given script file.

This script reads a file and computes the probability of each token according
to a language model, helping identify which tokens are surprising or unlikely.
"""

import argparse
import sys
from libvermcts import ModelWrapper


def compute_token_logits(model_wrapper: ModelWrapper, prompt: str, text: str):
    """
    Compute logits for each token in the text.

    Args:
        model_wrapper: The model wrapper to use for computing logits
        prompt: The context/prompt before the text (can be empty string)
        text: The text to analyze

    Returns:
        List of (token_text, probability, char_start, char_end) tuples
        where char_start and char_end are the character positions in the decoded text
    """
    print("Computing logits for each token...")

    # Tokenize the text
    tokens = model_wrapper.tokenizer.encode(text)

    token_logits = []
    char_pos = 0  # Track current character position in decoded text

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
            if prompt:
                prompt_tokens = model_wrapper.tokenizer.encode(prompt)
            else:
                prompt_tokens = []

            if context_tokens or prompt_tokens:
                input_ids = torch.tensor([prompt_tokens + context_tokens])
            else:
                # First token with no prompt - use empty context
                input_ids = torch.tensor([[model_wrapper.tokenizer.bos_token_id]]) if hasattr(model_wrapper.tokenizer, 'bos_token_id') and model_wrapper.tokenizer.bos_token_id is not None else torch.tensor([[]])

            if input_ids.shape[1] > 0:
                input_ids = input_ids.to(model_wrapper.device)

                # Get logits
                outputs = model_wrapper.model(input_ids)
                logits = outputs.logits[0, -1, :]

                # Convert to probabilities
                probs = torch.softmax(logits, dim=-1)

                # Get probability of the current token
                token_prob = probs[current_token].item()
            else:
                # No context, use uniform probability estimate
                token_prob = 0.0001

        # Decode the token to text
        token_text = model_wrapper.tokenizer.decode([current_token])
        token_length = len(token_text)

        # Store with character positions in the decoded text
        token_logits.append((token_text, token_prob, char_pos, char_pos + token_length))
        char_pos += token_length

        # Print progress every 100 tokens
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(tokens)} tokens...")

    return token_logits


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token probabilities in a script file using a language model"
    )
    parser.add_argument(
        "script_file",
        help="Path to the script file to analyze"
    )
    parser.add_argument(
        "--model",
        default="codellama/CodeLlama-7b-Instruct-hf",
        help="HuggingFace model to use (default: codellama/CodeLlama-7b-Instruct-hf)"
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Optional prompt/context to prepend before the script"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of lowest-probability tokens to display (default: 20)"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all tokens with their probabilities"
    )

    args = parser.parse_args()

    # Read the script file
    try:
        with open(args.script_file, 'r') as f:
            script_content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{args.script_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    print("=" * 70)
    print("Script Logit Analysis")
    print("=" * 70)
    print(f"File: {args.script_file}")
    print(f"Model: {args.model}")
    print(f"File size: {len(script_content)} characters")
    if args.prompt:
        print(f"Prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
    print("-" * 70)

    # Initialize model wrapper
    print("Loading model...")
    model_wrapper = ModelWrapper(args.model)
    print("Model loaded successfully.")
    print("-" * 70)

    # Compute logits for each token
    token_logits = compute_token_logits(model_wrapper, args.prompt, script_content)

    print(f"\nTotal tokens in file: {len(token_logits)}")

    # Calculate statistics
    probs = [prob for _, prob, _, _ in token_logits]
    avg_prob = sum(probs) / len(probs) if probs else 0
    min_prob = min(probs) if probs else 0
    max_prob = max(probs) if probs else 0

    # Calculate perplexity (geometric mean of inverse probabilities)
    import math
    log_probs = [math.log(p) if p > 0 else -100 for p in probs]
    avg_log_prob = sum(log_probs) / len(log_probs) if log_probs else -100
    perplexity = math.exp(-avg_log_prob)

    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    print(f"Average token probability: {avg_prob:.6f}")
    print(f"Min token probability: {min_prob:.8f}")
    print(f"Max token probability: {max_prob:.6f}")
    print(f"Perplexity: {perplexity:.2f}")

    # Sort by probability (ascending) to get lowest probabilities first
    sorted_tokens = sorted(token_logits, key=lambda x: x[1])

    # Display the N tokens with lowest probabilities
    print("\n" + "=" * 70)
    print(f"{args.top_n} Tokens with LOWEST Probabilities:")
    print("=" * 70)
    print(f"{'Rank':<6} {'Probability':<15} {'Char Range':<15} {'Token (repr)':<30}")
    print("-" * 70)
    for rank, (token_text, prob, char_start, char_end) in enumerate(sorted_tokens[:args.top_n], 1):
        token_repr = repr(token_text)[:30]  # Truncate long representations
        char_range = f"{char_start}-{char_end}"
        print(f"{rank:<6} {prob:<15.8f} {char_range:<15} {token_repr:<30}")

    # Optionally show all tokens
    if args.show_all:
        print("\n" + "=" * 70)
        print("All Tokens:")
        print("=" * 70)
        print(f"{'Char Range':<15} {'Probability':<15} {'Token (repr)':<40}")
        print("-" * 70)
        for token_text, prob, char_start, char_end in token_logits:
            token_repr = repr(token_text)[:40]
            char_range = f"{char_start}-{char_end}"
            print(f"{char_range:<15} {prob:<15.8f} {token_repr:<40}")

    # Reconstruct text with highlighted tokens
    print("\n" + "=" * 70)
    print("Script with Lowest Logit Tokens Highlighted")
    print("=" * 70)

    # Get set of character ranges to highlight (top N lowest probability tokens)
    highlight_ranges = set((char_start, char_end) for _, _, char_start, char_end in sorted_tokens[:args.top_n])

    # Terminal output with ANSI colors
    print("\n" + "-" * 70)
    print("Terminal Output (with red highlighting):")
    print("-" * 70)
    for token_text, prob, char_start, char_end in token_logits:
        if (char_start, char_end) in highlight_ranges:
            # ANSI red color code
            print(f"\033[91m{token_text}\033[0m", end='')
        else:
            print(token_text, end='')
    print()  # Final newline
    print("-" * 70)

    # LaTeX output
    print("\n" + "-" * 70)
    print("LaTeX Output:")
    print("-" * 70)
    print("% Add to your LaTeX preamble:")
    print("% \\usepackage{listings}")
    print("% \\usepackage{xcolor}")
    print("% \\lstset{basicstyle=\\ttfamily\\small, escapechar=|}")
    print("%")
    print("% Then use:")
    print()

    def escape_latex_in_lstlisting(text):
        """
        Escape text for use inside |\textcolor{...}{HERE}| in lstlisting.
        We need to escape LaTeX special characters.
        """
        # Backslash must be first
        text = text.replace('\\', '\\textbackslash{}')
        text = text.replace('{', '\\{')
        text = text.replace('}', '\\}')
        text = text.replace('_', '\\_')
        text = text.replace('%', '\\%')
        text = text.replace('&', '\\&')
        text = text.replace('#', '\\#')
        text = text.replace('$', '\\$')
        text = text.replace('^', '\\textasciicircum{}')
        text = text.replace('~', '\\textasciitilde{}')
        text = text.replace('|', '\\textbar{}')
        return text

    print("\\begin{lstlisting}")
    for token_text, prob, char_start, char_end in token_logits:
        if (char_start, char_end) in highlight_ranges:
            # Escape the token text for use in LaTeX
            escaped = escape_latex_in_lstlisting(token_text)
            print(f"|\\textcolor{{red}}{{{escaped}}}|", end='')
        else:
            # In lstlisting, most characters are literal, but we should
            # still be careful. The escape char | is handled above.
            print(token_text, end='')
    print()  # Final newline
    print("\\end{lstlisting}")
    print("-" * 70)

    print("=" * 70)


if __name__ == "__main__":
    main()
