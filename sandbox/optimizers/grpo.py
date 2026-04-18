"""
GRPO: Group Relative Policy Optimization for parameterized generators.

Adapted from DeepSeek's GRPO for non-differentiable reward settings.
Maintains a distribution over the parameter space, samples groups,
computes group-relative advantages (no value network), and shifts the
distribution toward high-reward regions.

For continuous params: Gaussian with advantage-weighted MLE updates.
For categorical params: softmax-weighted frequency updates.

The group-relative baseline is the key insight — advantages are normalized
within each generation, so no critic or baseline model is needed.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from sandbox.core import ExplorationTree, RunRecord


class GRPOOptimizer:
    """Group Relative Policy Optimization.

    Works best with generators that have explicit, bounded param spaces
    (e.g., ManimGenerator's pacing/density/style knobs).

    Args:
        param_space: Parameter space from generator.param_space()
        group_size: Number of samples per generation (G in GRPO)
        lr: Learning rate for distribution updates
        temperature: Softmax temperature for advantage weighting (lower = greedier)
        min_std_ratio: Minimum std as fraction of range (prevents collapse)
    """

    name = "grpo"

    def __init__(
        self,
        param_space: dict[str, dict[str, Any]],
        group_size: int = 8,
        lr: float = 0.3,
        temperature: float = 1.0,
        min_std_ratio: float = 0.05,
    ):
        self.param_space = param_space
        self.group_size = group_size
        self.lr = lr
        self.temperature = temperature
        self.min_std_ratio = min_std_ratio
        self.rng = np.random.default_rng()

        # Initialize distributions from param space
        self.distributions: dict[str, dict] = {}
        for name, spec in param_space.items():
            if spec["type"] == "float":
                lo, hi = spec["low"], spec["high"]
                self.distributions[name] = {
                    "type": "gaussian",
                    "mu": (lo + hi) / 2,
                    "sigma": (hi - lo) / 4,
                    "low": lo,
                    "high": hi,
                }
            elif spec["type"] == "categorical":
                choices = spec["choices"]
                n = len(choices)
                self.distributions[name] = {
                    "type": "categorical",
                    "probs": {c: 1.0 / n for c in choices},
                }

        self.history: list[dict] = []

    def suggest(
        self, tree: ExplorationTree, n: int | None = None
    ) -> list[dict[str, Any]]:
        """Sample a group of parameter configs from current distributions."""
        n = n or self.group_size
        samples = []

        for _ in range(n):
            params: dict[str, Any] = {}
            for name, dist in self.distributions.items():
                if dist["type"] == "gaussian":
                    val = self.rng.normal(dist["mu"], dist["sigma"])
                    val = float(np.clip(val, dist["low"], dist["high"]))
                    params[name] = val
                elif dist["type"] == "categorical":
                    choices = list(dist["probs"].keys())
                    probs = np.array([dist["probs"][c] for c in choices])
                    probs = probs / probs.sum()  # ensure normalized
                    params[name] = self.rng.choice(choices, p=probs)
            samples.append(params)

        return samples

    def update(
        self, tree: ExplorationTree, new_records: list[RunRecord]
    ) -> dict[str, Any]:
        """Update distributions using group-relative advantages.

        Core GRPO: advantages are (reward - group_mean) / group_std.
        No learned value function needed.
        """
        if not new_records:
            return {}

        rewards = np.array([r.reward for r in new_records])

        # -- Group-relative advantage computation (GRPO core) --
        if len(rewards) > 1 and rewards.std() > 1e-8:
            advantages = (rewards - rewards.mean()) / rewards.std()
        else:
            advantages = np.zeros_like(rewards)

        # Softmax weighting with temperature
        scaled = advantages / self.temperature
        scaled -= scaled.max()  # numerical stability
        weights = np.exp(scaled)
        weights = weights / weights.sum()

        # -- Update each distribution --
        for name, dist in self.distributions.items():
            if dist["type"] == "gaussian":
                values = np.array([r.params[name] for r in new_records])

                # Advantage-weighted MLE
                new_mu = float(np.sum(weights * values))
                new_sigma = float(
                    np.sqrt(np.sum(weights * (values - new_mu) ** 2))
                )

                # Enforce minimum sigma
                param_range = dist["high"] - dist["low"]
                min_sigma = param_range * self.min_std_ratio
                new_sigma = max(new_sigma, min_sigma)

                # Interpolate with current (momentum)
                dist["mu"] = dist["mu"] * (1 - self.lr) + new_mu * self.lr
                dist["sigma"] = (
                    dist["sigma"] * (1 - self.lr) + new_sigma * self.lr
                )

            elif dist["type"] == "categorical":
                # Accumulate advantage-weighted counts
                choice_advantage: dict[str, float] = {
                    c: 0.0 for c in dist["probs"]
                }
                for record, w in zip(new_records, weights):
                    choice = record.params[name]
                    choice_advantage[choice] += w

                # Blend with current probs
                for c in dist["probs"]:
                    dist["probs"][c] = (
                        dist["probs"][c] * (1 - self.lr)
                        + choice_advantage.get(c, 0) * self.lr
                    )

                # Renormalize
                total = sum(dist["probs"].values())
                if total > 0:
                    dist["probs"] = {
                        c: p / total for c, p in dist["probs"].items()
                    }

        # -- Diagnostics --
        diag = {
            "advantages": advantages.tolist(),
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "reward_max": float(rewards.max()),
            "distributions": self._serialize_distributions(),
        }
        self.history.append(diag)
        return diag

    def _serialize_distributions(self) -> dict:
        result = {}
        for name, dist in self.distributions.items():
            if dist["type"] == "gaussian":
                result[name] = {
                    "type": "gaussian",
                    "mu": round(dist["mu"], 4),
                    "sigma": round(dist["sigma"], 4),
                }
            elif dist["type"] == "categorical":
                result[name] = {
                    "type": "categorical",
                    "probs": {c: round(p, 4) for c, p in dist["probs"].items()},
                }
        return result

    def state_dict(self) -> dict:
        """Serialize optimizer state for saving/resuming."""
        return {
            "distributions": self.distributions,
            "history": self.history,
            "config": {
                "group_size": self.group_size,
                "lr": self.lr,
                "temperature": self.temperature,
                "min_std_ratio": self.min_std_ratio,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore optimizer state."""
        self.distributions = state["distributions"]
        self.history = state.get("history", [])
        config = state.get("config", {})
        self.group_size = config.get("group_size", self.group_size)
        self.lr = config.get("lr", self.lr)
        self.temperature = config.get("temperature", self.temperature)
