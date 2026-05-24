__all__ = [
    "AlphaFactorMiningWorkflow",
    "AgentLoopWorkflow",
    "SingleLoopWorkflow",
    "LoopPolicy",
    "RoundState",
    "LoopSession",
    "LoopRunner",
]


class _LazyExport:
    def __init__(self, module_path: str, attr_name: str) -> None:
        self.module_path = module_path
        self.attr_name = attr_name
        self._loaded = None

    def _load(self):
        if self._loaded is None:
            module = __import__(self.module_path, fromlist=[self.attr_name])
            self._loaded = getattr(module, self.attr_name)
        return self._loaded

    def __call__(self, *args, **kwargs):
        return self._load()(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._load(), name)

    def __repr__(self) -> str:
        return f"<lazy {self.module_path}.{self.attr_name}>"


_EXPORTS = {
    "AlphaFactorMiningWorkflow": ("workflows.alpha_factor_mining_workflow", "AlphaFactorMiningWorkflow"),
    "AgentLoopWorkflow": ("workflows.agent_loop_workflow", "AgentLoopWorkflow"),
    "SingleLoopWorkflow": ("workflows.single_loop_workflow", "SingleLoopWorkflow"),
    "LoopPolicy": ("workflows.loop_runtime", "LoopPolicy"),
    "RoundState": ("workflows.loop_runtime", "RoundState"),
    "LoopSession": ("workflows.loop_runtime", "LoopSession"),
    "LoopRunner": ("workflows.loop_runtime", "LoopRunner"),
}


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'workflows' has no attribute {name!r}")
    return _LazyExport(*target)
