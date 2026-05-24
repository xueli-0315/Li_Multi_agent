"""Shared trajectory pool helper for mining and evolution scripts."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class _Trajectory:
    trajectory_id: str
    round_idx: int
    phase: str = "original"
    hypothesis: str = ""
    factors: list[dict[str, Any]] = field(default_factory=list)
    backtest_result: Any = None
    backtest_metrics: dict[str, float | None] = field(default_factory=dict)
    feedback: str = ""
    parent_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    extra_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "round_idx": self.round_idx,
            "phase": self.phase,
            "hypothesis": self.hypothesis,
            "factors": self.factors,
            "backtest_result": self.backtest_result,
            "backtest_metrics": self.backtest_metrics,
            "feedback": self.feedback,
            "parent_ids": self.parent_ids,
            "created_at": self.created_at,
            "extra_info": self.extra_info,
        }


class TrajectoryPool:
    def __init__(self, save_path: Path | None = None) -> None:
        if save_path is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = _PROJECT_ROOT / "artifacts" / "trajectory_pool" / f"trajectory_pool_{run_id}.json"
        self.save_path = save_path
        self.trajectories: list[_Trajectory] = []
        self._load()

    def _load(self) -> None:
        if not self.save_path.exists():
            return
        try:
            data = json.loads(self.save_path.read_text(encoding="utf-8"))
            for item in data.get("trajectories", []):
                self.trajectories.append(_Trajectory(**{k: v for k, v in item.items() if k in _Trajectory.__dataclass_fields__}))
        except Exception:
            pass

    def add(self, trace_final_context: dict[str, Any], loop_index: int) -> None:
        hypothesis = str(trace_final_context.get("hypothesis", ""))
        experiment_spec = trace_final_context.get("experiment_spec", {})
        if isinstance(experiment_spec, dict):
            hypothesis = hypothesis or str(experiment_spec.get("target_hypothesis", ""))

        factors = []
        factor_impl = trace_final_context.get("factor_implementation", {})
        if isinstance(factor_impl, dict):
            factors = factor_impl.get("implementations", [])

        backtest_metrics = {}
        metrics = trace_final_context.get("metrics", {})
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                try:
                    backtest_metrics[str(k)] = float(v) if v is not None else None
                except (TypeError, ValueError):
                    backtest_metrics[str(k)] = None

        feedback_obj = trace_final_context.get("feedback", {})
        feedback_text = ""
        if isinstance(feedback_obj, dict):
            feedback_text = str(feedback_obj.get("summary", ""))

        trajectory_id = str(trace_final_context.get("trajectory_id", ""))
        if not trajectory_id:
            phase = str(trace_final_context.get("phase", "original"))
            trajectory_id = f"{phase}_{int(loop_index):03d}_{int(datetime.now().timestamp() % 1000):03d}"

        parent_ids = []
        raw_parents = trace_final_context.get("parent_ids", [])
        if isinstance(raw_parents, list):
            parent_ids = [str(b) for b in raw_parents]

        trajectory = _Trajectory(
            trajectory_id=trajectory_id,
            round_idx=loop_index,
            phase=str(trace_final_context.get("phase", "original")),
            hypothesis=hypothesis,
            factors=factors,
            backtest_result=trace_final_context.get("backtest_report"),
            backtest_metrics=backtest_metrics,
            feedback=feedback_text,
            parent_ids=parent_ids,
        )
        self.trajectories.append(trajectory)
        self._save()

    def _save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "trajectories": [t.to_dict() for t in self.trajectories],
            "save_path": str(self.save_path),
        }
        self.save_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
