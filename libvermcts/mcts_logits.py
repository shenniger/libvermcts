"""
MCTS-based verified text generation with logit information.
"""

from .verifier import VerifierResult
from .model import ModelWrapper
from .montecarlo.node import Node
from .montecarlo.montecarlo import MonteCarlo


def mcts_logits(prompt: str, verifier, model: str, max_tokens: int = 10000, expansion_count: int = 100, verbose: bool = False):
    """
    Monte Carlo Tree Search for verified text generation with logit probabilities.

    For MCTS, we ask the verifier about every new token. Responses:
    - BAD: Roll back based on MCTS.
    - GOOD: `add` is added as its own node in the tree.
    - DONE: We finish.
    - INCONCLUSIVE: Continue adding tokens to `add`.

    Args:
        prompt: Input prompt string
        verifier: Verifier function that takes (base, add, last_logit, logit_product) and returns VerifierResult
                  where last_logit is the probability (0.0-1.0) of the last generated token
                  and logit_product is the product of all token probabilities in add
        model: Model name string (e.g., "gpt-oss:20b")
        max_tokens: Maximum total number of tokens to generate (default: 10000)
        expansion_count: Number of MCTS expansions to perform (default: 100)
        verbose: If True, print all generation steps and verifier states (default: False)

    Returns:
        The verified generated text, or None if verification failed
    """
    # ANSI color codes
    GREEN = '\033[92m'  # Color for base text
    CYAN = '\033[96m'   # Color for newly generated text
    YELLOW = '\033[93m' # Color for logit information
    RESET = '\033[0m'

    # Initialize model wrapper
    model_wrapper = ModelWrapper(model)

    # Initialize MCTS with root node
    root_node = Node(prompt)
    montecarlo = MonteCarlo(root_node)

    tokens_used = 0
    expansion = 0

    def child_finder(node, mc):
        """
        Child finder function for MCTS expansion.

        This function generates new children for a node by:
        1. Generating tokens incrementally with their probabilities
        2. Verifying each addition with the verifier (passing the logit)
        3. Creating child nodes based on verification results
        """
        nonlocal tokens_used, expansion

        expansion += 1

        if verbose:
            print(f"\n{'='*60}")
            print(f"Expansion {expansion} (tokens used: {tokens_used}/{max_tokens})")
            print(f"{'='*60}")
            print(f"Current state: {GREEN}{node.state}{RESET}")
            print(f"{'-'*60}")

        if tokens_used >= max_tokens:
            if verbose:
                print(f"Max tokens reached, stopping expansion")
            return

        base = node.state
        add_tokens = []
        add_logits = []  # Track all logits
        last_logit = 0.0  # Track the logit of the last token
        logit_product = 1.0  # Product of all logits

        # Keep generating tokens until we get a clear verdict
        while tokens_used < max_tokens:
            # Generate one more token with logit information
            new_token_id, last_logit = model_wrapper.generate_one_token_with_logits(base, add_tokens)
            add_tokens.append(new_token_id)
            add_logits.append(last_logit)
            tokens_used += 1

            # Redecode the entire add sequence
            add = model_wrapper.decode_tokens(add_tokens)

            # Calculate the product of all logits
            logit_product = 1.0
            for logit in add_logits:
                logit_product *= logit

            # Verify the addition (passing the logit of the last token and the product)
            result = verifier(base, add, last_logit, logit_product)

            if result == VerifierResult.DONE:
                # Success! Create a child with the complete generation
                child = Node(base + add)
                node.add_child(child)
                child.update_win_value(1)
                child.update_policy_value(1)

                if verbose:
                    print(f"\nFull text: {GREEN}{base}{RESET}{CYAN}{add}{RESET}")
                    print(f"Last logit: {YELLOW}{last_logit:.4f}{RESET}, Product: {YELLOW}{logit_product:.6e}{RESET}")
                    print(f"Verifier result: {result.value}")
                    print(f"DONE - Solution found!")

                # Mark as solution
                mc.solution = base + add
                return

            elif result == VerifierResult.GOOD:
                # This addition is good but incomplete
                # Create a child node with base + add
                child = Node(base + add)
                node.add_child(child)
                child.update_win_value(0.5)
                child.update_policy_value(1)

                # Also create a sibling that stays at current state (for exploration)
                sibling = Node(base)
                node.add_child(sibling)
                sibling.update_policy_value(0.2)

                if verbose:
                    print(f"\nFull text: {GREEN}{base}{RESET}{CYAN}{add}{RESET}")
                    print(f"Last logit: {YELLOW}{last_logit:.4f}{RESET}, Product: {YELLOW}{logit_product:.6e}{RESET}")
                    print(f"Verifier result: {result.value}")
                    print(f"GOOD - Creating child node")

                return

            elif result == VerifierResult.BAD:
                # This is bad, mark node as bad and return
                node.update_win_value(-1)

                if verbose:
                    print(f"\nFull text: {GREEN}{base}{RESET}{CYAN}{add}{RESET}")
                    print(f"Last logit: {YELLOW}{last_logit:.4f}{RESET}, Product: {YELLOW}{logit_product:.6e}{RESET}")
                    print(f"Verifier result: {result.value}")
                    print(f"BAD - Rolling back")

                return

            elif result == VerifierResult.INCONCLUSIVE:
                # Keep adding tokens, show status on one line that overwrites itself
                if verbose:
                    # Escape newlines in the text for single-line display
                    base_display = base.replace('\n', '\\n')
                    add_display = add.replace('\n', '\\n')
                    print(f"\rINCONCLUSIVE (last={YELLOW}{last_logit:.4f}{RESET}, prod={YELLOW}{logit_product:.6e}{RESET}) - {GREEN}{base_display}{RESET}{CYAN}{add_display}{RESET}", end='', flush=True)
                continue

        # Ran out of tokens
        if verbose:
            print(f"Ran out of tokens in this expansion")
        return

    # Set the child finder
    montecarlo.child_finder = child_finder

    # Run MCTS simulation
    montecarlo.simulate(expansion_count)

    # Return the solution if found
    return montecarlo.solution
