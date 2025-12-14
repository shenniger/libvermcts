"""
MCTS-based verified text generation.
"""

from .verifier import VerifierResult
from .model import ModelWrapper
from .montecarlo.node import Node
from .montecarlo.montecarlo import MonteCarlo


def mcts(prompt: str, verifier, model: str, max_tokens: int = 10000, expansion_count: int = 100, verbose: bool = False):
    """
    Monte Carlo Tree Search for verified text generation.

    For MCTS, we ask the verifier about every new token. Responses:
    - BAD: Roll back based on MCTS.
    - GOOD: `add` is added as its own node in the tree.
    - DONE: We finish.
    - INCONCLUSIVE: Continue adding tokens to `add`.

    Args:
        prompt: Input prompt string
        verifier: Verifier function that takes (base, add) and returns VerifierResult
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
        1. Generating tokens incrementally
        2. Verifying each addition with the verifier
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
        add = ""

        # Keep generating tokens until we get a clear verdict
        while tokens_used < max_tokens:
            # Generate one more token
            new_token = model_wrapper.generate_one_token(base + add)
            add += new_token
            tokens_used += 1

            # Verify the addition
            result = verifier(base, add)

            if result == VerifierResult.DONE:
                # Success! Create a child with the complete generation
                child = Node(base + add)
                node.add_child(child)
                child.update_win_value(1)
                child.update_policy_value(1)

                if verbose:
                    print(f"\nFull text: {GREEN}{base}{RESET}{CYAN}{add}{RESET}")
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
                    print(f"Verifier result: {result.value}")
                    print(f"GOOD - Creating child node")

                return

            elif result == VerifierResult.BAD:
                # This is bad, mark node as bad and return
                node.update_win_value(-1)

                if verbose:
                    print(f"\nFull text: {GREEN}{base}{RESET}{CYAN}{add}{RESET}")
                    print(f"Verifier result: {result.value}")
                    print(f"BAD - Rolling back")

                return

            elif result == VerifierResult.INCONCLUSIVE:
                # Keep adding tokens, show status on one line that overwrites itself
                if verbose:
                    # Escape newlines in the text for single-line display
                    base_display = base.replace('\n', '\\n')
                    add_display = add.replace('\n', '\\n')
                    print(f"\rINCONCLUSIVE - {GREEN}{base_display}{RESET}{CYAN}{add_display}{RESET}", end='', flush=True)
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
