"""
Verifier module for defining verifier function contracts and result types.
"""

from enum import Enum


class VerifierResult(Enum):
    """
    Result types for verifier functions.

    A verifier function takes two strings:
    - base: The last successfully verified value
    - add: The new content to verify

    And returns one of these values:
    """
    BAD = "BAD"  # Definitely wrong. Need to roll back.
    GOOD = "GOOD"  # `add` looks good, but is still incomplete. Can append to base.
    DONE = "DONE"  # Generation complete. We have all we want.
    INCONCLUSIVE = "INCONCLUSIVE"  # No news on whether `add` is good. Need more tokens.
