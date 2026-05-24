from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Literal

from core import ContextStore, Orchestrator
from factor_runtime.factor_library_manager import FactorLibraryManager
from factor_runtime.factor_quality_gate import FactorQualityGate, QualityGateConfig
from infra.structured_logger import StructuredLogger
from schemas import LoopTrace, QlibFactorExperiment, RoundPhase, SharedContext, StrategyTrajectory, TrajectoryPool


@dataclass(frozen=True)
class LoopPolicy:
    max_loops: int
    stop_on_error: bool = True
    retry_per_round: int = 0


@dataclass
class RoundState:
    loop_index: int
    attempt_index: int
    phase: str
    trajectory_id: str
    parent_ids: list[str]
    input_shared_keys: list[str]
    output_shared_keys: list[str]
    agent_count: int
    error_count: int
    errors: list[str]
    started_at: datetime
    ended_at: datetime


@dataclass
class LoopSession:
    policy: LoopPolicy
    rounds: list[RoundState] = field(default_factory=list)
    loop_traces: list[LoopTrace] = field(default_factory=list)
    latest_shared_context: dict[str, Any] = field(default_factory=dict)
    status: Literal["running", "completed", "failed"] = "running"
    total_errors: int = 0
    private_context_snapshots: list[dict[str, dict[str, Any]]] = field(default_factory=list)
    trajectory_pool: TrajectoryPool = field(default_factory=TrajectoryPool)
    factor_library_path: str = ""


class LoopRunner:
    def __init__(
        self,
        *,
        orchestrator: Orchestrator,
        context_store: ContextStore | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        quality_gate: FactorQualityGate | None = None,
        logger: StructuredLogger | None = None,
        post_round_callback: Callable[[LoopTrace, int], None] | None = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.context_store = context_store
        self.progress_callback = progress_callback
        self.quality_gate = quality_gate or FactorQualityGate()
        self.logger = logger or StructuredLogger()
        self.post_round_callback = post_round_callback

    def _notify_progress(self, payload: dict[str, object]) -> None:
        if self.progress_callback is None:
            return
        try:
            self.progress_callback(payload)
        except Exception:
            return

    @staticmethod
    def _extract_errors(trace: LoopTrace) -> list[str]:
        return [str(record.error) for record in trace.records if record.error]

    @staticmethod
    def _extract_private_snapshot(context_store: ContextStore | None, trace: LoopTrace) -> dict[str, dict[str, Any]]:
        if context_store is None:
            return {}
        snapshot: dict[str, dict[str, Any]] = {}
        for record in trace.records:
            agent_context = context_store.get_private_context(record.agent_name)
            snapshot[record.agent_name] = dict(agent_context.payload)
        return snapshot

    @staticmethod
    def _build_round_payload(*, loop_index: int, shared_payload: dict[str, Any]) -> dict[str, Any]:
        current = dict(shared_payload)
        phase = str(current.get("phase", RoundPhase.ORIGINAL.value))
        if phase not in {item.value for item in RoundPhase}:
            phase = RoundPhase.ORIGINAL.value
        round_idx = max(loop_index, int(current.get("round_idx", loop_index) or loop_index))
        parent_ids = [str(item) for item in list(current.get("parent_ids", []))]
        trajectory_id = str(current.get("trajectory_id", "")).strip()
        expected_prefix = f"{phase}_{int(round_idx):03d}_"
        if not trajectory_id or not trajectory_id.startswith(expected_prefix):
            trajectory_id = StrategyTrajectory.generate_id(direction_id=0, round_idx=round_idx, phase=RoundPhase(phase))
        current["phase"] = phase
        current["round_idx"] = round_idx
        current["parent_ids"] = parent_ids
        current["trajectory_id"] = trajectory_id
        return current

    @staticmethod
    def _trajectory_state_path() -> str:
        return str(FactorLibraryManager.project_root() / "artifacts" / "trajectory_pool.json")

    @staticmethod
    def _factor_library_path() -> str:
        return str(FactorLibraryManager.default_library_path())

    @staticmethod
    def _record_trajectory(session: LoopSession, payload: dict[str, Any]) -> None:
        experiment = QlibFactorExperiment.from_shared_payload(payload)
        if not experiment.trajectory_id:
            return
        phase_text = str(experiment.phase or RoundPhase.ORIGINAL.value)
        if phase_text not in {item.value for item in RoundPhase}:
            phase_text = RoundPhase.ORIGINAL.value
        metric_payload: dict[str, float | None] = {}
        for key, value in experiment.metrics.items():
            try:
                metric_payload[str(key)] = float(value)
            except Exception:
                metric_payload[str(key)] = None
        trajectory = StrategyTrajectory(
            trajectory_id=experiment.trajectory_id,
            direction_id=0,
            round_idx=int(experiment.round_idx),
            phase=RoundPhase(phase_text),
            hypothesis=experiment.target_hypothesis,
            factors=[item.to_dict() for item in experiment.factors],
            backtest_result=experiment.backtest_report,
            backtest_metrics=metric_payload,
            feedback=str(experiment.feedback.get("summary", "")),
            feedback_details=dict(experiment.feedback),
            parent_ids=list(experiment.parent_ids),
            extra_info={"experiment_id": experiment.experiment_id},
        )
        if session.trajectory_pool.get_by_id(trajectory.trajectory_id) is None:
            session.trajectory_pool.add(trajectory)

    def run(
        self,
        *,
        initial_shared_payload: dict[str, object] | None = None,
        policy: LoopPolicy,
    ) -> LoopSession:
        session = LoopSession(
            policy=policy,
            latest_shared_context=dict(initial_shared_payload or {}),
            trajectory_pool=TrajectoryPool(save_path=self._trajectory_state_path()),
            factor_library_path=self._factor_library_path(),
        )
        session.trajectory_pool.load()
        factor_library = FactorLibraryManager(session.factor_library_path)
        shared_payload = dict(initial_shared_payload or {})
        total_loops = max(1, int(policy.max_loops))
        for loop_index in range(1, total_loops + 1):
            attempt = 0
            while True:
                shared_payload = self._build_round_payload(loop_index=loop_index, shared_payload=shared_payload)
                started_at = datetime.now()
                trace = self.orchestrator.run_one_loop(SharedContext(payload=dict(shared_payload)))
                ended_at = datetime.now()
                errors = self._extract_errors(trace)
                phase = str(shared_payload.get("phase", RoundPhase.ORIGINAL.value))
                trajectory_id = str(shared_payload.get("trajectory_id", ""))
                parent_ids = [str(item) for item in list(shared_payload.get("parent_ids", []))]
                round_state = RoundState(
                    loop_index=loop_index,
                    attempt_index=attempt,
                    phase=phase,
                    trajectory_id=trajectory_id,
                    parent_ids=parent_ids,
                    input_shared_keys=sorted(shared_payload.keys()),
                    output_shared_keys=sorted(trace.final_shared_context.keys()),
                    agent_count=len(trace.records),
                    error_count=len(errors),
                    errors=errors,
                    started_at=started_at,
                    ended_at=ended_at,
                )
                session.rounds.append(round_state)
                session.loop_traces.append(trace)
                session.latest_shared_context = dict(trace.final_shared_context)
                session.private_context_snapshots.append(self._extract_private_snapshot(self.context_store, trace))
                session.total_errors += round_state.error_count
                self._record_trajectory(session, trace.final_shared_context)
                # --- Step-4: Optional Post-Round Hook ---
                if self.post_round_callback:
                    self.post_round_callback(trace, loop_index)
                
                self._notify_progress(
                    {
                        "stage": "round_finished",
                        "loop_index": loop_index,
                        "attempt_index": attempt,
                        "error_count": round_state.error_count,
                        "agent_count": round_state.agent_count,
                    }
                )
                if round_state.error_count == 0:
                    shared_payload = dict(trace.final_shared_context)
                    break
                if policy.stop_on_error:
                    session.status = "failed"
                    return session
                if attempt >= max(0, int(policy.retry_per_round)):
                    shared_payload = dict(trace.final_shared_context)
                    break
                attempt += 1
        session.status = "completed"
        return session
