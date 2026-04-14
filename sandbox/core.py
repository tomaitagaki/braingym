"""
Sandbox core: data structures and orchestration for brain-reward optimization.

RunRecord: single evaluation (params -> video -> brain -> reward)
ExplorationTree: tracks branching exploration history
BrainReward: configurable reward from cortical network signals
SandboxRunner: generator -> Tribe -> reward -> optimizer loop
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


# ============================================================
# RunRecord
# ============================================================

@dataclass
class RunRecord:
    """Single evaluation: params -> video -> brain -> reward."""

    id: str
    params: dict[str, Any]
    video_path: str | None = None
    reward: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    brain_signals: dict[str, list[float]] = field(default_factory=dict)
    parent_id: str | None = None
    generation: int = 0
    timestamp: float = field(default_factory=time.time)
    generator_name: str = ""
    optimizer_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        # Drop large timeseries from serialization by default
        d.pop("brain_signals", None)
        return d

    def to_full_dict(self) -> dict:
        return asdict(self)


# ============================================================
# ExplorationTree
# ============================================================

class ExplorationTree:
    """Tree of exploration runs with branching, history, and persistence."""

    def __init__(self, save_dir: Path):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.records: dict[str, RunRecord] = {}
        self._load()

    # -- mutation --

    def add(self, record: RunRecord) -> None:
        self.records[record.id] = record
        self._save_record(record)

    # -- queries --

    def get(self, run_id: str) -> RunRecord:
        return self.records[run_id]

    def children(self, run_id: str) -> list[RunRecord]:
        return [r for r in self.records.values() if r.parent_id == run_id]

    def ancestors(self, run_id: str) -> list[RunRecord]:
        chain: list[RunRecord] = []
        current = self.records.get(run_id)
        while current and current.parent_id:
            parent = self.records.get(current.parent_id)
            if parent:
                chain.append(parent)
            current = parent
        return chain

    def generation(self, gen: int) -> list[RunRecord]:
        return [r for r in self.records.values() if r.generation == gen]

    def best(self, n: int = 5) -> list[RunRecord]:
        scored = [r for r in self.records.values() if r.reward is not None]
        scored.sort(key=lambda r: r.reward, reverse=True)
        return scored[:n]

    def all_records(self) -> list[RunRecord]:
        return sorted(self.records.values(), key=lambda r: r.timestamp)

    @property
    def max_generation(self) -> int:
        if not self.records:
            return -1
        return max(r.generation for r in self.records.values())

    def __len__(self) -> int:
        return len(self.records)

    # -- persistence --

    def _save_record(self, record: RunRecord) -> None:
        path = self.save_dir / f"{record.id}.json"
        path.write_text(json.dumps(record.to_full_dict(), indent=2, default=str))

    def _load(self) -> None:
        for path in self.save_dir.glob("*.json"):
            if path.name == "index.json":
                continue
            try:
                data = json.loads(path.read_text())
                self.records[data["id"]] = RunRecord(**data)
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

    def save_index(self) -> None:
        """Save a summary index of all runs."""
        index = [r.to_dict() for r in self.all_records()]
        path = self.save_dir / "index.json"
        path.write_text(json.dumps(index, indent=2, default=str))


# ============================================================
# BrainReward
# ============================================================

class BrainReward:
    """Configurable reward function from cortical network signals.

    Extracts cognitive metrics from Tribe's 7-network signals and
    computes a weighted scalar reward. Also returns per-metric breakdown
    for multi-objective optimization (GEPA Pareto front).
    """

    DEFAULT_WEIGHTS = {
        "engagement": 0.4,      # -DMN (less mind-wandering)
        "attention": 0.3,       # DAN - DMN
        "salience": 0.2,        # VAN (surprise/novelty)
        "emotional": 0.1,       # Limbic
    }

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def __call__(
        self, signals: dict[str, np.ndarray]
    ) -> tuple[float, dict[str, float]]:
        """Compute reward from network signals.

        Args:
            signals: {network_name: (n_timesteps,) array} from Tribe

        Returns:
            (scalar_reward, metrics_dict)
        """
        metrics = self.extract_metrics(signals)
        reward = sum(self.weights.get(k, 0) * metrics[k] for k in self.weights)
        return float(reward), metrics

    def extract_metrics(self, signals: dict[str, np.ndarray]) -> dict[str, float]:
        dan = np.asarray(signals["DAN"])
        dmn = np.asarray(signals["DMN"])
        van = np.asarray(signals["VAN"])
        limbic = np.asarray(signals["Limbic"])
        fpn = np.asarray(signals["FPN"])

        n3 = min(3, len(dan))  # first 3 seconds

        return {
            # Core metrics (used in default reward)
            "engagement": float(-np.mean(dmn)),
            "attention": float(np.mean(dan) - np.mean(dmn)),
            "salience": float(np.mean(van)),
            "emotional": float(np.mean(limbic)),
            # Temporal metrics (useful as GEPA objectives)
            "hook_3s": float(np.mean(dan[:n3]) - np.mean(dmn[:n3])),
            "sustained": float(-np.std(dmn)),
            "peak_salience": float(np.max(van)),
            "cognitive_load": float(np.mean(fpn) + np.mean(dan) - np.mean(dmn)),
            # Per-network means
            "net_DAN": float(np.mean(dan)),
            "net_DMN": float(np.mean(dmn)),
            "net_VAN": float(np.mean(van)),
            "net_FPN": float(np.mean(fpn)),
            "net_Limbic": float(np.mean(limbic)),
            "net_Vis": float(np.mean(signals.get("Vis", [0]))),
            "net_SomMot": float(np.mean(signals.get("SomMot", [0]))),
        }


# ============================================================
# Pareto utilities (for GEPA multi-objective)
# ============================================================

def is_dominated(obj_a: list[float], obj_b: list[float]) -> bool:
    """True if obj_a is dominated by obj_b (b >= a on all, b > a on at least one)."""
    return (
        all(b >= a for a, b in zip(obj_a, obj_b))
        and any(b > a for a, b in zip(obj_a, obj_b))
    )


def pareto_front(
    records: list[RunRecord], objectives: list[str]
) -> list[RunRecord]:
    """Return non-dominated records for the given objectives (all maximized)."""
    if not records:
        return []

    obj_vectors = []
    for r in records:
        obj_vectors.append([r.metrics.get(o, 0.0) for o in objectives])

    dominated = set()
    for i in range(len(records)):
        for j in range(len(records)):
            if i != j and is_dominated(obj_vectors[i], obj_vectors[j]):
                dominated.add(i)
                break

    return [records[i] for i in range(len(records)) if i not in dominated]


# ============================================================
# SandboxRunner
# ============================================================

class SandboxRunner:
    """Orchestrates the generator -> Tribe -> reward -> optimizer loop."""

    def __init__(
        self,
        generator,   # Generator protocol
        scorer,      # TribeScorer
        reward: BrainReward,
        optimizer,   # Optimizer protocol
        save_dir: Path | str,
    ):
        self.generator = generator
        self.scorer = scorer
        self.reward = reward
        self.optimizer = optimizer
        self.tree = ExplorationTree(Path(save_dir) / "runs")
        self.video_dir = Path(save_dir) / "videos"
        self.video_dir.mkdir(parents=True, exist_ok=True)
        self.generation = self.tree.max_generation + 1

    def step(self, n_samples: int = 4) -> tuple[list[RunRecord], dict]:
        """Run one optimization step: suggest -> generate -> score -> update.

        Returns:
            (records, optimizer_diagnostics)
        """
        # 1. Get parameter suggestions
        param_batch = self.optimizer.suggest(self.tree, n=n_samples)

        # 2. Generate + score each
        records: list[RunRecord] = []
        for i, params in enumerate(param_batch):
            run_id = f"gen{self.generation:03d}_{i:02d}_{uuid.uuid4().hex[:6]}"
            output_path = self.video_dir / f"{run_id}.mp4"

            # Generate video
            video_path = self.generator.generate(params, output_path)

            # Score with Tribe
            preds = self.scorer.predict(video_path)
            signals = self.scorer.extract_signals(preds)
            reward_val, metrics = self.reward(signals)

            record = RunRecord(
                id=run_id,
                params=params,
                video_path=str(video_path),
                reward=reward_val,
                metrics=metrics,
                brain_signals={k: v.tolist() for k, v in signals.items()},
                generation=self.generation,
                generator_name=self.generator.name,
                optimizer_name=self.optimizer.name,
                metadata=params.get("_metadata", {}),
            )
            records.append(record)
            self.tree.add(record)

            print(
                f"  [{run_id}] reward={reward_val:.4f}  "
                f"attn={metrics.get('attention', 0):.3f}  "
                f"engage={metrics.get('engagement', 0):.3f}"
            )

        # 3. Update optimizer
        diagnostics = self.optimizer.update(self.tree, records)

        # 4. Print generation summary
        rewards = [r.reward for r in records]
        best = max(records, key=lambda r: r.reward)
        print(
            f"\n  Gen {self.generation}: "
            f"best={max(rewards):.4f}  mean={np.mean(rewards):.4f}  "
            f"std={np.std(rewards):.4f}  "
            f"best_params={best.params}"
        )

        self.generation += 1
        self.tree.save_index()
        return records, diagnostics

    def run(
        self, n_steps: int = 10, n_samples: int = 4, early_stop: float | None = None
    ) -> ExplorationTree:
        """Run multiple optimization steps.

        Args:
            n_steps: Number of generations
            n_samples: Samples per generation
            early_stop: Stop if best reward exceeds this value
        """
        print(
            f"Sandbox: {self.generator.name} x {self.optimizer.name}  "
            f"({n_steps} steps x {n_samples} samples)\n"
        )

        for step in range(n_steps):
            print(f"--- Step {step + 1}/{n_steps} ---")
            records, diagnostics = self.step(n_samples)

            if early_stop is not None:
                best_ever = self.tree.best(1)
                if best_ever and best_ever[0].reward >= early_stop:
                    print(f"\nEarly stop: reward {best_ever[0].reward:.4f} >= {early_stop}")
                    break

        # Final summary
        print("\n" + "=" * 60)
        print("EXPLORATION COMPLETE")
        print("=" * 60)
        top5 = self.tree.best(5)
        for i, r in enumerate(top5):
            print(f"  #{i+1}: reward={r.reward:.4f}  gen={r.generation}  params={r.params}")

        return self.tree
