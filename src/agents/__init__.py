__all__ = [
    "HypothesisAgentV2",
    "ExperimentDesignerAgent",
    "FactorCoderAgent",
    "BacktestRunnerAgent",
    "FeedbackSummarizerAgent",
    "ReportReaderOrganizerAgent",
]


def __getattr__(name: str):
    if name == "BacktestRunnerAgent":
        from agents.backtest_runner_agent import BacktestRunnerAgent

        return BacktestRunnerAgent
    if name == "FeedbackSummarizerAgent":
        from agents.feedback import FeedbackSummarizerAgent

        return FeedbackSummarizerAgent
    if name == "HypothesisAgentV2":
        from agents.hypothesis import HypothesisAgentV2

        return HypothesisAgentV2
    if name == "ExperimentDesignerAgent":
        from agents.experiment_designer_agent import ExperimentDesignerAgent

        return ExperimentDesignerAgent
    if name == "FactorCoderAgent":
        from agents.factor_coder_agent import FactorCoderAgent

        return FactorCoderAgent
    if name == "ReportReaderOrganizerAgent":
        from agents.report_reader_organizer_agent import ReportReaderOrganizerAgent

        return ReportReaderOrganizerAgent
    raise AttributeError(f"module 'trading_agents.agents' has no attribute {name!r}")
