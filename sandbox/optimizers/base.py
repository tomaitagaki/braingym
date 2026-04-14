"""Optimizer protocol: suggests parameters and learns from brain rewards."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from sandbox.core import ExplorationTree, RunRecord


@runtime_checkable
class Optimizer(Protocol):
    """Interface for optimization strategies.

    Suggest: propose a batch of parameter configs to try.
    Update: learn from the scored results.

    Implementations: GRPOOptimizer (distribution-based), GEPAOptimizer (LLM-guided).
    """

    name: str

    def suggest(
        self, tree: ExplorationTree, n: int = 4
    ) -> list[dict[str, Any]]:
        """Suggest next batch of parameters to evaluate.

        Args:
            tree: Full exploration history
            n: Number of suggestions

        Returns:
            List of parameter dicts
        """
        ...

    def update(
        self, tree: ExplorationTree, new_records: list[RunRecord]
    ) -> dict[str, Any]:
        """Update internal state after new evaluations.

        Returns:
            Optimizer diagnostics (distribution stats, advantages, etc.)
        """
        ...
