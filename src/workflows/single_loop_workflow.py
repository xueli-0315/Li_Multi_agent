from __future__ import annotations

from schemas import LoopTrace
from workflows.alpha_factor_mining_workflow import AlphaFactorMiningWorkflow


class SingleLoopWorkflow:
    def __init__(self) -> None:
        self.workflow = AlphaFactorMiningWorkflow()

    def run(self, initial_shared_payload: dict[str, object] | None = None) -> LoopTrace:
        return self.workflow.run(initial_shared_payload)
