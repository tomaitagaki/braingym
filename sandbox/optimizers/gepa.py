"""
GEPA: Genetic-Pareto Prompt Optimization via DSPy.

From https://arxiv.org/abs/2507.19457 — optimizes content strategies through
natural language reflection instead of gradient updates.

The loop:
1. Maintain a population of content "strategies" (natural language)
2. Each strategy → concrete generator params → video → Tribe → brain reward
3. LLM reflects on what distinguished high vs low performers
4. Crossover/mutate strategies guided by reflection
5. Track Pareto front for multi-objective optimization

Uses DSPy for structured LLM calls: reflection, crossover, strategy→params.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

try:
    import dspy
except ImportError:
    dspy = None

from sandbox.core import ExplorationTree, RunRecord, pareto_front


# ============================================================
# DSPy Signatures (defined lazily to avoid import error when dspy absent)
# ============================================================

def _check_dspy():
    if dspy is None:
        raise ImportError(
            "GEPA requires dspy-ai. Install with: pip install dspy-ai"
        )


def _build_signatures():
    """Define DSPy Signature classes. Called once at GEPAOptimizer init."""

    class GenerateStrategy(dspy.Signature):
        """Generate a content presentation strategy optimized for cognitive engagement.

        Given a concept to teach and brain-based objectives, propose a specific
        strategy for how to present the content (pacing, visuals, narrative, etc.)
        that would maximize the target brain metrics.
        """
        concept: str = dspy.InputField(desc="The concept to present in the video")
        objectives: str = dspy.InputField(
            desc="Brain metrics to optimize (e.g., engagement, attention, salience)"
        )
        existing_strategies: str = dspy.InputField(
            desc="Strategies already tried (avoid duplicates)"
        )
        strategy: str = dspy.OutputField(
            desc="A specific, detailed content presentation strategy"
        )

    class ReflectOnResults(dspy.Signature):
        """Analyze brain-scored content variations to understand what drives
        cognitive engagement. Compare high-performing and low-performing
        variations to extract actionable insights.
        """
        top_results: str = dspy.InputField(
            desc="Top-performing variations with brain metrics and strategies"
        )
        bottom_results: str = dspy.InputField(
            desc="Bottom-performing variations with brain metrics and strategies"
        )
        insights: str = dspy.OutputField(
            desc="What specifically made top variations more cognitively engaging"
        )

    class CrossoverStrategies(dspy.Signature):
        """Combine two successful content strategies into a new one, guided by
        insights about what drives cognitive engagement. The child should
        inherit the best elements of both parents.
        """
        parent_a: str = dspy.InputField(
            desc="First parent strategy with its brain reward scores"
        )
        parent_b: str = dspy.InputField(
            desc="Second parent strategy with its brain reward scores"
        )
        reflection: str = dspy.InputField(
            desc="Insights on what drives cognitive engagement"
        )
        child_strategy: str = dspy.OutputField(
            desc="New strategy combining best elements of both parents"
        )

    class MutateStrategy(dspy.Signature):
        """Modify a content strategy to explore a new direction. Make a meaningful
        change — not just rewording, but a substantive shift in approach.
        """
        strategy: str = dspy.InputField(desc="Strategy to mutate")
        reflection: str = dspy.InputField(desc="Insights on what works/doesn't")
        direction: str = dspy.InputField(
            desc="What aspect to explore (e.g., 'try faster pacing', 'add more visuals')"
        )
        mutated_strategy: str = dspy.OutputField(desc="Modified strategy")

    class StrategyToParams(dspy.Signature):
        """Convert a natural language content strategy into concrete generator
        parameters. The parameters must be valid values within the given
        parameter space.
        """
        strategy: str = dspy.InputField(desc="Content presentation strategy")
        param_space: str = dspy.InputField(
            desc="Available parameters with types and valid ranges/choices"
        )
        params_json: str = dspy.OutputField(
            desc="JSON object mapping parameter names to values"
        )

    return (
        GenerateStrategy, ReflectOnResults, CrossoverStrategies,
        MutateStrategy, StrategyToParams,
    )


# ============================================================
# GEPA Optimizer
# ============================================================

class GEPAOptimizer:
    """Genetic-Pareto Prompt Optimization using DSPy.

    Unlike GRPO which operates on explicit parameter distributions,
    GEPA evolves natural language *strategies* and uses an LLM to
    convert them to concrete parameters. This makes it:
    - Model-agnostic (works with any generator)
    - Interpretable (strategies are human-readable)
    - Multi-objective (Pareto front, not just scalar reward)

    Args:
        param_space: From generator.param_space()
        concept: What the content is about
        population_size: Strategies per generation
        objectives: Brain metrics for Pareto ranking
        mutation_rate: Fraction of children that are mutations vs crossovers
        model: LLM model for DSPy
    """

    name = "gepa"

    def __init__(
        self,
        param_space: dict[str, dict[str, Any]],
        concept: str,
        population_size: int = 8,
        objectives: list[str] | None = None,
        mutation_rate: float = 0.3,
        model: str = "anthropic/claude-sonnet-4-20250514",
    ):
        _check_dspy()

        self.param_space = param_space
        self.concept = concept
        self.pop_size = population_size
        self.objectives = objectives or ["engagement", "attention", "salience"]
        self.mutation_rate = mutation_rate
        self.rng = np.random.default_rng()

        # DSPy setup
        self.lm = dspy.LM(model)
        dspy.configure(lm=self.lm)

        # Build DSPy signatures and modules
        sigs = _build_signatures()
        (GenerateStrategy, ReflectOnResults, CrossoverStrategies,
         MutateStrategy, StrategyToParams) = sigs

        self.generate_strategy = dspy.ChainOfThought(GenerateStrategy)
        self.reflect = dspy.ChainOfThought(ReflectOnResults)
        self.crossover = dspy.ChainOfThought(CrossoverStrategies)
        self.mutate = dspy.ChainOfThought(MutateStrategy)
        self.to_params = dspy.ChainOfThought(StrategyToParams)

        # State
        self.strategies: dict[str, str] = {}  # run_id -> strategy text
        self.insights_history: list[str] = []
        self._pareto: list[RunRecord] = []

    def suggest(
        self, tree: ExplorationTree, n: int | None = None
    ) -> list[dict[str, Any]]:
        n = n or self.pop_size

        if not tree.records:
            return self._initial_population(n)

        # 1. Reflect on what worked
        insights = self._do_reflect(tree)
        self.insights_history.append(insights)

        # 2. Update Pareto front
        scored = [r for r in tree.records.values() if r.reward is not None]
        self._pareto = pareto_front(scored, self.objectives)

        # 3. Generate next generation via crossover + mutation
        return self._evolve(n, insights)

    def update(
        self, tree: ExplorationTree, new_records: list[RunRecord]
    ) -> dict[str, Any]:
        """Store strategy associations. Actual learning happens in suggest()."""
        # Associate records with their strategies (stored in metadata)
        for r in new_records:
            strategy = r.metadata.get("strategy", "")
            if strategy:
                self.strategies[r.id] = strategy

        scored = [r for r in tree.records.values() if r.reward is not None]
        self._pareto = pareto_front(scored, self.objectives)

        return {
            "pareto_size": len(self._pareto),
            "n_strategies": len(self.strategies),
            "n_insights": len(self.insights_history),
            "pareto_rewards": [r.reward for r in self._pareto],
        }

    # -- internal --

    def _initial_population(self, n: int) -> list[dict[str, Any]]:
        """Generate diverse initial strategies."""
        param_batch = []
        generated_strategies: list[str] = []

        for i in range(n):
            result = self.generate_strategy(
                concept=self.concept,
                objectives=", ".join(self.objectives),
                existing_strategies=(
                    "\n".join(f"- {s}" for s in generated_strategies)
                    if generated_strategies
                    else "None yet — be diverse"
                ),
            )
            strategy = result.strategy
            generated_strategies.append(strategy)

            params = self._strategy_to_params(strategy)
            params["_metadata"] = {"strategy": strategy, "origin": "initial"}
            param_batch.append(params)

        return param_batch

    def _evolve(self, n: int, insights: str) -> list[dict[str, Any]]:
        """Create next generation via crossover and mutation."""
        # Select parents from Pareto front + top performers
        parents = self._select_parents()
        if not parents:
            return self._initial_population(n)

        param_batch = []
        for i in range(n):
            if self.rng.random() < self.mutation_rate:
                # Mutation: perturb a single parent
                parent = parents[i % len(parents)]
                params = self._do_mutate(parent, insights)
            else:
                # Crossover: combine two parents
                pa = parents[i % len(parents)]
                pb = parents[(i + 1) % len(parents)]
                params = self._do_crossover(pa, pb, insights)

            param_batch.append(params)

        return param_batch

    def _select_parents(self) -> list[tuple[str, RunRecord]]:
        """Select parent (strategy, record) pairs from Pareto front."""
        parents = []
        for r in self._pareto:
            strategy = self.strategies.get(r.id, r.metadata.get("strategy", ""))
            if strategy:
                parents.append((strategy, r))

        # If Pareto front is small, add top-reward records too
        if len(parents) < 2:
            for r in sorted(
                [r for r in self.strategies if isinstance(r, str)],
                key=lambda rid: (
                    self._pareto[0].reward if self._pareto else 0
                ),
            ):
                pass  # fallback not needed if pareto has entries

        return parents if parents else []

    def _do_reflect(self, tree: ExplorationTree) -> str:
        """Reflect on top vs bottom performers."""
        scored = sorted(
            [r for r in tree.records.values() if r.reward is not None],
            key=lambda r: r.reward,
            reverse=True,
        )
        if len(scored) < 2:
            return "Not enough data to reflect yet."

        n = max(2, len(scored) // 3)
        top = scored[:n]
        bottom = scored[-n:]

        def format_record(r: RunRecord) -> str:
            strategy = self.strategies.get(r.id, r.metadata.get("strategy", "N/A"))
            obj_str = ", ".join(
                f"{o}={r.metrics.get(o, 0):.3f}" for o in self.objectives
            )
            return f"Reward={r.reward:.4f} [{obj_str}] Strategy: {strategy}"

        top_str = "\n".join(f"- {format_record(r)}" for r in top)
        bottom_str = "\n".join(f"- {format_record(r)}" for r in bottom)

        result = self.reflect(top_results=top_str, bottom_results=bottom_str)
        return result.insights

    def _do_crossover(
        self,
        parent_a: tuple[str, RunRecord],
        parent_b: tuple[str, RunRecord],
        insights: str,
    ) -> dict[str, Any]:
        """Crossover two parent strategies."""
        sa, ra = parent_a
        sb, rb = parent_b

        result = self.crossover(
            parent_a=f"Strategy: {sa}\nReward: {ra.reward:.4f}\nMetrics: {ra.metrics}",
            parent_b=f"Strategy: {sb}\nReward: {rb.reward:.4f}\nMetrics: {rb.metrics}",
            reflection=insights,
        )

        strategy = result.child_strategy
        params = self._strategy_to_params(strategy)
        params["_metadata"] = {
            "strategy": strategy,
            "origin": "crossover",
            "parents": [ra.id, rb.id],
        }
        return params

    def _do_mutate(
        self,
        parent: tuple[str, RunRecord],
        insights: str,
    ) -> dict[str, Any]:
        """Mutate a parent strategy."""
        strategy, record = parent

        # Pick a random exploration direction
        directions = [
            "try faster pacing with quicker transitions",
            "add more visual elements and geometric demonstrations",
            "reduce text and let visuals speak",
            "add a stronger narrative hook at the beginning",
            "try a surprise opening that defies expectations",
            "slow down and give more time for each concept to sink in",
            "increase emotional engagement through storytelling",
            "make it more concise and information-dense",
        ]
        direction = self.rng.choice(directions)

        result = self.mutate(
            strategy=strategy,
            reflection=insights,
            direction=direction,
        )

        new_strategy = result.mutated_strategy
        params = self._strategy_to_params(new_strategy)
        params["_metadata"] = {
            "strategy": new_strategy,
            "origin": "mutation",
            "parent": record.id,
            "direction": direction,
        }
        return params

    def _strategy_to_params(self, strategy: str) -> dict[str, Any]:
        """Use LLM to convert strategy text to concrete parameters."""
        space_desc = json.dumps(
            {
                name: {k: v for k, v in spec.items() if k != "desc"}
                for name, spec in self.param_space.items()
            },
            indent=2,
        )

        result = self.to_params(
            strategy=strategy,
            param_space=space_desc,
        )

        try:
            # Extract JSON from response (handle markdown code blocks)
            raw = result.params_json
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            params = json.loads(raw)

            # Validate and clamp
            return self._validate_params(params)
        except (json.JSONDecodeError, KeyError, ValueError):
            # Fallback: random params
            return self._random_params()

    def _validate_params(self, params: dict) -> dict[str, Any]:
        """Ensure params are within bounds."""
        validated: dict[str, Any] = {}
        for name, spec in self.param_space.items():
            if name not in params:
                validated[name] = self._random_param(name, spec)
                continue

            val = params[name]
            if spec["type"] == "float":
                val = float(val)
                val = max(spec["low"], min(spec["high"], val))
                validated[name] = val
            elif spec["type"] == "categorical":
                if val not in spec["choices"]:
                    val = self.rng.choice(spec["choices"])
                validated[name] = val

        # Preserve metadata
        if "_metadata" in params:
            validated["_metadata"] = params["_metadata"]

        return validated

    def _random_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, spec in self.param_space.items():
            params[name] = self._random_param(name, spec)
        return params

    def _random_param(self, name: str, spec: dict) -> Any:
        if spec["type"] == "float":
            return float(self.rng.uniform(spec["low"], spec["high"]))
        elif spec["type"] == "categorical":
            return self.rng.choice(spec["choices"])

    def state_dict(self) -> dict:
        return {
            "strategies": self.strategies,
            "insights_history": self.insights_history,
            "pareto_ids": [r.id for r in self._pareto],
        }
