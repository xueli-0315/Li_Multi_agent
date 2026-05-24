from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


@dataclass
class AgentContext:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class SharedContext:
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Hypothesis:
    statement: str
    assumptions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    steps: list[str]
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Execution:
    status: str
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Feedback:
    summary: str
    score: float
    actions: list[str] = field(default_factory=list)


@dataclass
class AgentResult:
    shared_updates: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceRecord:
    agent_name: str
    started_at: datetime
    ended_at: datetime
    input_snapshot: dict[str, Any]
    output_snapshot: dict[str, Any]
    error: str | None = None


@dataclass
class LoopTrace:
    records: list[TraceRecord] = field(default_factory=list)
    final_shared_context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentFactorTask:
    factor_name: str
    description: str = ""
    formulation: str = ""
    expression: str = ""
    variables: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentFactorTask":
        return cls(
            factor_name=str(payload.get("factor_name", payload.get("name", ""))).strip(),
            description=str(payload.get("description", payload.get("factor_description", ""))),
            formulation=str(payload.get("formulation", payload.get("factor_formulation", ""))),
            expression=str(payload.get("expression", payload.get("factor_expression", ""))),
            variables=dict(payload.get("variables", {})) if isinstance(payload.get("variables", {}), dict) else {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "description": self.description,
            "formulation": self.formulation,
            "expression": self.expression,
            "variables": dict(self.variables),
        }


@dataclass
class QlibFactorExperiment:
    experiment_id: str
    target_hypothesis: str = ""
    factors: list[ExperimentFactorTask] = field(default_factory=list)
    factor_names: list[str] = field(default_factory=list)
    deduplicated_against: list[str] = field(default_factory=list)
    task_plan: list[str] = field(default_factory=list)
    attempt_count: int = 0
    consistency_gate: dict[str, Any] = field(default_factory=dict)
    factor_implementation: dict[str, Any] = field(default_factory=dict)
    calculation_report: dict[str, Any] = field(default_factory=dict)
    backtest_report: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    feedback: dict[str, Any] = field(default_factory=dict)
    phase: str = "original"
    round_idx: int = 0
    trajectory_id: str = ""
    parent_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    extra: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id(*, round_idx: int, phase: str, seed: str = "") -> str:
        safe_seed = "".join(ch for ch in str(seed) if ch.isalnum() or ch in {"_", "-"}).strip("_-")
        suffix = safe_seed or "exp"
        return f"{phase.lower()}_{int(round_idx):03d}_{suffix}"

    @classmethod
    def from_shared_payload(cls, payload: dict[str, Any]) -> "QlibFactorExperiment":
        raw = payload.get("qlib_factor_experiment")
        if isinstance(raw, dict):
            return cls.from_dict(raw)
        factor_rows = payload.get("experiment_spec", {}).get("factors", []) if isinstance(payload.get("experiment_spec", {}), dict) else []
        factors = [ExperimentFactorTask.from_dict(item) for item in factor_rows if isinstance(item, dict)]
        experiment_spec = payload.get("experiment_spec", {})
        experiment_id = str(payload.get("experiment_id", "")).strip() or cls.generate_id(
            round_idx=int(payload.get("round_idx", 0) or 0),
            phase=str(payload.get("phase", "original")),
            seed=str(payload.get("trajectory_id", "")),
        )
        return cls(
            experiment_id=experiment_id,
            target_hypothesis=str(experiment_spec.get("target_hypothesis", "")) if isinstance(experiment_spec, dict) else "",
            factors=factors,
            factor_names=[task.factor_name for task in factors],
            deduplicated_against=list(experiment_spec.get("deduplicated_against", [])) if isinstance(experiment_spec, dict) else [],
            task_plan=list(payload.get("task_plan", [])),
            attempt_count=int(experiment_spec.get("attempt_count", 0)) if isinstance(experiment_spec, dict) else 0,
            consistency_gate=dict(experiment_spec.get("consistency_gate", {})) if isinstance(experiment_spec, dict) else {},
            factor_implementation=dict(payload.get("factor_implementation", {})),
            calculation_report=dict(payload.get("calculation_report", {})),
            backtest_report=dict(payload.get("backtest_report", {})),
            metrics={str(k): float(v) for k, v in dict(payload.get("metrics", {})).items() if isinstance(v, (int, float))},
            feedback=dict(payload.get("feedback", {})),
            phase=str(payload.get("phase", "original")),
            round_idx=int(payload.get("round_idx", 0) or 0),
            trajectory_id=str(payload.get("trajectory_id", "")),
            parent_ids=[str(item) for item in list(payload.get("parent_ids", []))],
            extra={},
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QlibFactorExperiment":
        factors = payload.get("factors", [])
        return cls(
            experiment_id=str(payload.get("experiment_id", "")),
            target_hypothesis=str(payload.get("target_hypothesis", "")),
            factors=[ExperimentFactorTask.from_dict(item) for item in factors if isinstance(item, dict)],
            factor_names=[str(item) for item in list(payload.get("factor_names", []))],
            deduplicated_against=[str(item) for item in list(payload.get("deduplicated_against", []))],
            task_plan=[str(item) for item in list(payload.get("task_plan", []))],
            attempt_count=int(payload.get("attempt_count", 0) or 0),
            consistency_gate=dict(payload.get("consistency_gate", {})),
            factor_implementation=dict(payload.get("factor_implementation", {})),
            calculation_report=dict(payload.get("calculation_report", {})),
            backtest_report=dict(payload.get("backtest_report", {})),
            metrics={str(k): float(v) for k, v in dict(payload.get("metrics", {})).items() if isinstance(v, (int, float))},
            feedback=dict(payload.get("feedback", {})),
            phase=str(payload.get("phase", "original")),
            round_idx=int(payload.get("round_idx", 0) or 0),
            trajectory_id=str(payload.get("trajectory_id", "")),
            parent_ids=[str(item) for item in list(payload.get("parent_ids", []))],
            created_at=str(payload.get("created_at", datetime.now().isoformat())),
            updated_at=str(payload.get("updated_at", datetime.now().isoformat())),
            extra=dict(payload.get("extra", {})),
        )

    def to_experiment_spec(self) -> dict[str, Any]:
        return {
            "target_hypothesis": self.target_hypothesis,
            "factors": [task.to_dict() for task in self.factors],
            "factor_names": list(self.factor_names or [task.factor_name for task in self.factors]),
            "deduplicated_against": list(self.deduplicated_against),
            "attempt_count": int(self.attempt_count),
            "consistency_gate": dict(self.consistency_gate),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "target_hypothesis": self.target_hypothesis,
            "factors": [task.to_dict() for task in self.factors],
            "factor_names": list(self.factor_names or [task.factor_name for task in self.factors]),
            "deduplicated_against": list(self.deduplicated_against),
            "task_plan": list(self.task_plan),
            "attempt_count": int(self.attempt_count),
            "consistency_gate": dict(self.consistency_gate),
            "factor_implementation": dict(self.factor_implementation),
            "calculation_report": dict(self.calculation_report),
            "backtest_report": dict(self.backtest_report),
            "metrics": {str(k): float(v) for k, v in self.metrics.items()},
            "feedback": dict(self.feedback),
            "phase": self.phase,
            "round_idx": int(self.round_idx),
            "trajectory_id": self.trajectory_id,
            "parent_ids": list(self.parent_ids),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "extra": dict(self.extra),
        }

    def update_timestamps(self) -> None:
        self.updated_at = datetime.now().isoformat()


class RoundPhase(str, Enum):
    ORIGINAL = "original"
    MUTATION = "mutation"
    CROSSOVER = "crossover"


@dataclass
class StrategyTrajectory:
    trajectory_id: str
    direction_id: int
    round_idx: int
    phase: RoundPhase
    hypothesis: str = ""
    hypothesis_details: dict[str, Any] = field(default_factory=dict)
    factors: list[dict[str, Any]] = field(default_factory=list)
    backtest_result: Any = None
    backtest_metrics: dict[str, float | None] = field(default_factory=dict)
    feedback: str = ""
    feedback_details: dict[str, Any] = field(default_factory=dict)
    parent_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    extra_info: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def generate_id(direction_id: int, round_idx: int, phase: RoundPhase) -> str:
        return f"{phase.value}_{int(round_idx):03d}_{int(direction_id):03d}"

    def get_primary_metric(self) -> float | None:
        for key in ("RankIC", "Rank IC", "IC", "information_ratio", "annualized_return"):
            value = self.backtest_metrics.get(key)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    continue
        return None

    def is_successful(self) -> bool:
        metric = self.get_primary_metric()
        return metric is not None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "trajectory_id": self.trajectory_id,
            "direction_id": self.direction_id,
            "round_idx": self.round_idx,
            "phase": self.phase.value,
            "hypothesis": self.hypothesis,
            "hypothesis_details": self.hypothesis_details,
            "factors": self.factors,
            "backtest_result": self.backtest_result,
            "backtest_metrics": self.backtest_metrics,
            "feedback": self.feedback,
            "feedback_details": self.feedback_details,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at,
            "extra_info": self.extra_info,
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyTrajectory":
        return cls(
            trajectory_id=str(payload.get("trajectory_id", "")),
            direction_id=int(payload.get("direction_id", 0) or 0),
            round_idx=int(payload.get("round_idx", 0) or 0),
            phase=RoundPhase(str(payload.get("phase", RoundPhase.ORIGINAL.value))),
            hypothesis=str(payload.get("hypothesis", "")),
            hypothesis_details=dict(payload.get("hypothesis_details", {})),
            factors=list(payload.get("factors", [])),
            backtest_result=payload.get("backtest_result"),
            backtest_metrics=dict(payload.get("backtest_metrics", {})),
            feedback=str(payload.get("feedback", "")),
            feedback_details=dict(payload.get("feedback_details", {})),
            parent_ids=[str(item) for item in list(payload.get("parent_ids", []))],
            created_at=str(payload.get("created_at", datetime.now().isoformat())),
            extra_info=dict(payload.get("extra_info", {})),
        )


@dataclass
class TrajectoryPool:
    trajectories: list[StrategyTrajectory] = field(default_factory=list)
    save_path: str | None = None

    def add(self, trajectory: StrategyTrajectory) -> None:
        self.trajectories.append(trajectory)
        self._save()

    def get_all(self) -> list[StrategyTrajectory]:
        return list(self.trajectories)

    def get_by_phase(self, phase: RoundPhase) -> list[StrategyTrajectory]:
        return [item for item in self.trajectories if item.phase == phase]

    def get_by_id(self, trajectory_id: str) -> StrategyTrajectory | None:
        for item in self.trajectories:
            if item.trajectory_id == trajectory_id:
                return item
        return None

    def get_statistics(self) -> dict[str, Any]:
        phase_counts = {
            RoundPhase.ORIGINAL.value: len(self.get_by_phase(RoundPhase.ORIGINAL)),
            RoundPhase.MUTATION.value: len(self.get_by_phase(RoundPhase.MUTATION)),
            RoundPhase.CROSSOVER.value: len(self.get_by_phase(RoundPhase.CROSSOVER)),
        }
        return {
            "total": len(self.trajectories),
            "by_phase": phase_counts,
        }

    def get_best_trajectories(self, top_n: int = 5) -> list[StrategyTrajectory]:
        valid = [item for item in self.trajectories if item.is_successful()]
        valid.sort(key=lambda item: item.get_primary_metric() or 0.0, reverse=True)
        return valid[: max(0, int(top_n))]

    def select_parents_for_mutation(self, top_n: int = 3) -> list[StrategyTrajectory]:
        raise NotImplementedError("TODO: mutation parent selection will be enabled with evolution execution")

    def select_groups_for_crossover(self, group_size: int = 2, n_groups: int = 3) -> list[list[StrategyTrajectory]]:
        raise NotImplementedError("TODO: crossover grouping will be enabled with evolution execution")

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectories": [item.to_dict() for item in self.trajectories],
            "save_path": self._to_portable_path(self.save_path),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrajectoryPool":
        items = payload.get("trajectories", [])
        return cls(
            trajectories=[StrategyTrajectory.from_dict(item) for item in items if isinstance(item, dict)],
            save_path=str(payload.get("save_path", "")) or None,
        )

    def load(self) -> None:
        if not self.save_path:
            return
        path = self._resolve_save_path()
        if not path.exists():
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = self.from_dict(data)
        self.trajectories = loaded.trajectories

    def _save(self) -> None:
        if not self.save_path:
            return
        path = self._resolve_save_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _to_portable_path(cls, raw_path: str | None) -> str:
        text = str(raw_path or "").strip()
        if not text:
            return ""
        path = Path(text)
        if not path.is_absolute():
            return path.as_posix()
        try:
            return path.resolve().relative_to(cls._project_root().resolve()).as_posix()
        except Exception:
            return path.as_posix()

    def _resolve_save_path(self) -> Path:
        text = str(self.save_path or "").strip()
        path = Path(text)
        if path.is_absolute():
            return path
        return self._project_root() / path
