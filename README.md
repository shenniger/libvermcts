# libvermcts

A library with a standardized interface for strategies to use LLMs with external signals.

## Installation

```bash
pip install -r requirements.txt
```

## Structure

- `libvermcts/` - Main library directory
  - `verifier.py` - Verifier result types and contracts
  - `model.py` - PyTorch model wrapper for text generation
  - `wholesampling.py` - Whole sampling implementation
  - `mcts.py` - Monte Carlo Tree Search implementation
  - `montecarlo/` - MCTS node and search tree implementation

- `example_simple.py` - Simple example with basic text verification
- `example_code_generation.py` - Code generation example with syntax verification
- `example_mcts.py` - MCTS-based generation with backtracking

## Usage

### Basic Wholesampling

```python
from libvermcts import wholesampling, VerifierResult

def my_verifier(base: str, add: str) -> VerifierResult:
    text = base + add
    # Your verification logic here
    if is_complete(text):
        return VerifierResult.DONE
    elif is_bad(text):
        return VerifierResult.BAD
    else:
        return VerifierResult.INCONCLUSIVE

result = wholesampling(
    prompt="Your prompt here",
    verifier=my_verifier,
    model="google/gemma-3-1b-it",
    max_tokens=10000,
    verbose=False  # Set to True to see all generation attempts
)
```

### Verifier Function Contract

A verifier function takes two strings:
- `base`: The last successfully verified value
- `add`: The new content to verify

And returns one of:
- `VerifierResult.BAD` - Definitely wrong, need to roll back
- `VerifierResult.GOOD` - Looks good but incomplete (can append to base)
- `VerifierResult.DONE` - Generation complete
- `VerifierResult.INCONCLUSIVE` - No news, need more tokens

### MCTS (Monte Carlo Tree Search)

```python
from libvermcts import mcts, VerifierResult

def my_verifier(base: str, add: str) -> VerifierResult:
    text = base + add
    # Your verification logic here
    if is_complete(text):
        return VerifierResult.DONE
    elif is_bad(text):
        return VerifierResult.BAD
    elif is_good_progress(text):
        return VerifierResult.GOOD
    else:
        return VerifierResult.INCONCLUSIVE

result = mcts(
    prompt="Your prompt here",
    verifier=my_verifier,
    model="google/gemma-3-1b-it",
    max_tokens=10000,
    expansion_count=100,
    verbose=False  # Set to True to see all expansions and verifier states
)
```

**MCTS vs Wholesampling:**
- **Wholesampling**: Generates complete outputs and verifies them as a whole. Faster but cannot recover from mistakes mid-generation.
- **MCTS**: Verifies each token incrementally and can backtrack when the verifier returns `BAD`. Slower but more robust for complex constraints.

## Examples

Run the examples:

```bash
python example_simple.py
python example_code_generation.py
python example_mcts.py
```
