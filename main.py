from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and ((value[0] == '"' and value[-1] == '"') or (value[0] == "'" and value[-1] == "'")):
        value = value[1:-1]
    return key, value


def _load_dotenv_files() -> None:
    for env_file in [PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"]:
        if not env_file.exists() or not env_file.is_file():
            continue
        for line in env_file.read_text(encoding="utf-8").splitlines():
            parsed = _parse_env_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if key not in os.environ:
                os.environ[key] = value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _truncate_text(value: str, limit: int = 260) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}...(truncated)"


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _summarize_llm_request(request: dict[str, Any]) -> dict[str, Any]:
    messages = request.get("messages", [])
    message_summaries: list[dict[str, Any]] = []
    if isinstance(messages, list):
        for item in messages[:4]:
            if not isinstance(item, dict):
                continue
            content_text = _content_to_text(item.get("content", ""))
            message_summaries.append(
                {
                    "role": str(item.get("role", "")),
                    "chars": len(content_text),
                    "preview": _truncate_text(content_text),
                }
            )
    return {
        "model": request.get("model"),
        "json_mode": request.get("json_mode"),
        "max_tokens": request.get("max_tokens"),
        "temperature": request.get("temperature"),
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "messages": message_summaries,
    }


def _summarize_llm_response(response: dict[str, Any]) -> dict[str, Any]:
    content_text = _content_to_text(response.get("content", ""))
    raw = response.get("raw", {})
    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    return {
        "finish_reason": response.get("finish_reason"),
        "content_chars": len(content_text),
        "content_preview": _truncate_text(content_text),
        "usage": usage if isinstance(usage, dict) else {},
    }


def _summarize_factor_steps(steps: Any) -> list[dict[str, Any]]:
    if not isinstance(steps, list):
        return []
    summarized: list[dict[str, Any]] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        code_value = item.get("code", "")
        if isinstance(code_value, list):
            code_lines = len(code_value)
            code_preview = _truncate_text("\n".join(str(line) for line in code_value[:8]))
        else:
            code_text = str(code_value)
            code_lines = code_text.count("\n") + (1 if code_text else 0)
            code_preview = _truncate_text(code_text)
        summarized.append(
            {
                "loop_index": item.get("loop_index"),
                "factor_name": item.get("factor_name"),
                "attempt_index": item.get("attempt_index"),
                "source": item.get("source"),
                "expression": item.get("expression"),
                "acceptable": item.get("acceptable"),
                "code_lines": code_lines,
                "code_preview": code_preview,
            }
        )
    return summarized


def _summarize_private_context(private_context: Any, *, keep_full_factor_coder_steps: bool = False) -> dict[str, Any]:
    if not isinstance(private_context, dict):
        return {}
    summary: dict[str, Any] = {
        "keys": sorted(str(key) for key in private_context.keys()),
    }
    recent_trace = private_context.get("recent_trace")
    if isinstance(recent_trace, list):
        summary["recent_trace_count"] = len(recent_trace)
    loop_trace = private_context.get("factor_coder_loop_trace")
    if isinstance(loop_trace, list):
        summary["factor_coder_loop_trace_count"] = len(loop_trace)
    generated_steps = private_context.get("factor_coder_generated_code_steps")
    if isinstance(generated_steps, list):
        summary["factor_coder_generated_code_steps"] = (
            generated_steps if keep_full_factor_coder_steps else _summarize_factor_steps(generated_steps)
        )
    state_snapshot = private_context.get("factor_coder_state_snapshot")
    if isinstance(state_snapshot, list):
        if keep_full_factor_coder_steps:
            summary["factor_coder_state_snapshot"] = state_snapshot
        else:
            summary["factor_coder_state_snapshot"] = [
                {
                    "factor_name": item.get("factor_name"),
                    "final_decision": item.get("final_decision"),
                    "dropped": item.get("dropped"),
                    "failed_attempts": item.get("failed_attempts"),
                    "attempt_count": item.get("attempt_count"),
                }
                for item in state_snapshot
                if isinstance(item, dict)
            ]
    return summary


def _summarize_shared_updates(shared_updates: Any) -> dict[str, Any]:
    if not isinstance(shared_updates, dict):
        return {}
    summary: dict[str, Any] = {"keys": sorted(str(key) for key in shared_updates.keys())}
    calculation_report = shared_updates.get("calculation_report")
    if isinstance(calculation_report, dict):
        summary["calculation_report"] = dict(calculation_report)
    backtest_report = shared_updates.get("backtest_report")
    if isinstance(backtest_report, dict):
        summary["backtest_report"] = dict(backtest_report)
    factor_implementation = shared_updates.get("factor_implementation")
    if isinstance(factor_implementation, dict):
        implementations = factor_implementation.get("implementations", [])
        impl_summary: list[dict[str, Any]] = []
        if isinstance(implementations, list):
            for item in implementations:
                if not isinstance(item, dict):
                    continue
                impl_summary.append(
                    {
                        "factor_name": item.get("factor_name"),
                        "status": item.get("status"),
                        "attempts": item.get("attempts"),
                        "expression": item.get("expression"),
                    }
                )
        summary["factor_implementation"] = {
            "implementation_count": factor_implementation.get("implementation_count"),
            "implementations": impl_summary,
        }
    return summary


def _summarize_snapshot(
    snapshot: Any,
    *,
    is_input: bool,
    keep_full_factor_coder_steps: bool = False,
) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        return {}
    result: dict[str, Any] = {}
    private_context = snapshot.get("private_context")
    if isinstance(private_context, dict):
        result["private_context"] = _summarize_private_context(
            private_context,
            keep_full_factor_coder_steps=keep_full_factor_coder_steps,
        )
    shared_context = snapshot.get("shared_context")
    if isinstance(shared_context, dict):
        result["shared_context_keys"] = sorted(str(key) for key in shared_context.keys())
    if not is_input:
        artifacts = snapshot.get("artifacts")
        if isinstance(artifacts, dict):
            result["artifacts_keys"] = sorted(str(key) for key in artifacts.keys())
        shared_updates = snapshot.get("shared_updates")
        if isinstance(shared_updates, dict):
            result["shared_updates"] = _summarize_shared_updates(shared_updates)
    return result


def _compact_workflow_payload(payload: dict[str, Any]) -> dict[str, Any]:
    keep_full_factor_coder_steps = str(payload.get("agent_name", "")) == "factor_coder_agent"
    compacted: dict[str, Any] = {
        "stage": payload.get("stage"),
        "agent_name": payload.get("agent_name"),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "failed_at": payload.get("failed_at"),
        "error": payload.get("error"),
    }
    if "loop_index" in payload:
        compacted["loop_index"] = payload.get("loop_index")
    if "attempt_index" in payload:
        compacted["attempt_index"] = payload.get("attempt_index")
    if "phase" in payload:
        compacted["phase"] = payload.get("phase")
    if "error_count" in payload:
        compacted["error_count"] = payload.get("error_count")
    input_snapshot = payload.get("input_snapshot")
    if isinstance(input_snapshot, dict):
        compacted["input_snapshot"] = _summarize_snapshot(
            input_snapshot,
            is_input=True,
            keep_full_factor_coder_steps=keep_full_factor_coder_steps,
        )
    output_snapshot = payload.get("output_snapshot")
    if isinstance(output_snapshot, dict):
        compacted["output_snapshot"] = _summarize_snapshot(
            output_snapshot,
            is_input=False,
            keep_full_factor_coder_steps=keep_full_factor_coder_steps,
        )
    return {key: value for key, value in compacted.items() if value is not None}


def _compact_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    compacted: dict[str, Any] = {
        "loop_index": payload.get("loop_index"),
        "agent_name": payload.get("agent_name"),
        "phase": payload.get("phase"),
        "attempt": payload.get("attempt"),
        "next_attempt": payload.get("next_attempt"),
        "max_attempts": payload.get("max_attempts"),
        "continue_round": payload.get("continue_round"),
        "wait_seconds": payload.get("wait_seconds"),
        "provider_elapsed_ms": payload.get("provider_elapsed_ms"),
        "error": payload.get("error"),
    }
    request = payload.get("request")
    if isinstance(request, dict):
        compacted["request"] = _summarize_llm_request(request)
    response = payload.get("response")
    if isinstance(response, dict):
        compacted["response"] = _summarize_llm_response(response)
    return {key: value for key, value in compacted.items() if value is not None}


def _compact_debug_event(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload", {})
    if not isinstance(payload, dict):
        return event
    tag = str(event.get("tag", ""))
    compacted_payload = payload
    if tag == "workflow_event":
        compacted_payload = _compact_workflow_payload(payload)
    elif tag == "llm_io":
        compacted_payload = _compact_llm_payload(payload)
    return {
        "tag": event.get("tag"),
        "timestamp": event.get("timestamp"),
        "payload": compacted_payload,
    }


def _append_pretty_log(log_path: Path | None, event: dict[str, Any]) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(_json_safe(event), ensure_ascii=False, indent=2)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(serialized)
        file.write("\n\n")


def _build_debug_log_path(enabled: bool) -> Path | None:
    if not enabled:
        return None
    log_dir = PROJECT_ROOT / "artifacts" / "debug_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = f"debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    return log_dir / filename


def _build_debug_logger(
    enabled: bool,
    tag: str,
    log_path: Path | None = None,
) -> Callable[[dict[str, Any]], None] | None:
    """仅当 --debug 启用时返回 logger（写 stderr + 合并的 debug log）。"""
    if not enabled:
        return None

    def _logger(payload: dict[str, Any]) -> None:
        event = {
            "tag": tag,
            "timestamp": datetime.now().isoformat(),
            "payload": payload,
        }
        compacted_event = _compact_debug_event(event)
        print(json.dumps(_json_safe(compacted_event), ensure_ascii=False), file=sys.stderr, flush=True)
        _append_pretty_log(log_path, compacted_event)

    return _logger


def _build_jsonl_logger(
    log_path: Path,
    tag_filter: str | None = None,
) -> Callable[[dict[str, Any]], None]:
    """Always-on JSONL logger: write compacted events to a specific file."""
    def _logger(payload: dict[str, Any]) -> None:
        event = {
            "tag": tag_filter or "event",
            "timestamp": datetime.now().isoformat(),
            "payload": payload,
        }
        compacted_event = _compact_debug_event(event)
        line = json.dumps(_json_safe(compacted_event), ensure_ascii=False)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return _logger


def _chain_loggers(*loggers: Callable[[dict[str, Any]], None] | None) -> Callable[[dict[str, Any]], None]:
    active = [fn for fn in loggers if fn is not None]
    def _combined(payload: dict[str, Any]) -> None:
        for fn in active:
            fn(payload)
    return _combined


def _prepare_data_bundle(
    *,
    panel_data_path: str,
    text_data_path: str = "",
    log_base: Path,
    write_data_artifacts: bool = False,
    debug_symbol_count: int = 20,
    debug_time_steps: int = 180,
):
    from adapters import UnifiedMarketDataAdapter

    if not text_data_path and not write_data_artifacts:
        return Path(panel_data_path), None

    bundle = UnifiedMarketDataAdapter(
        panel_data_path,
        text_data_path=text_data_path or None,
        artifact_dir=log_base / "data_bundle",
        write_artifacts=True,
        debug_symbol_count=debug_symbol_count,
        debug_time_steps=debug_time_steps,
    ).load()
    merged_path = Path(bundle.artifacts.get("merged_panel", panel_data_path))
    print(f"Prepared data bundle: {log_base / 'data_bundle'}")
    return merged_path, bundle


def _run_evolution(
    *,
    panel_data_path: str,
    text_data_path: str,
    write_data_artifacts: bool,
    debug_symbol_count: int,
    debug_time_steps: int,
    debug_enabled: bool,
    log_base: Path,
    progress_logger: Callable[[dict[str, Any]], None],
    raw_io_logger: Callable[[dict[str, Any]], None],
    evolve_target: str = "factor",
    factor_ga_mode: str = "hybrid",
    num_generations: int = 5,
    population_size: int = 10,
    seed: int = 42,
) -> None:
    """调用 scripts/run_factor_evolution.py 的进化流程。"""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.run_factor_evolution import EvolutionConfig, run_evolution

    print(f"Running evolution mode, logs: {log_base}")

    config = EvolutionConfig()
    config.panel_data_path = Path(panel_data_path)
    config.text_data_path = Path(text_data_path) if text_data_path else None
    config.write_data_artifacts = bool(write_data_artifacts)
    config.debug_symbol_count = debug_symbol_count
    config.debug_time_steps = debug_time_steps
    config.evolve_target = evolve_target
    config.ga_mode = factor_ga_mode
    config.factor_ga_mode = factor_ga_mode
    config.num_generations = num_generations
    config.population_size = population_size
    config.seed = seed
    config.final_audit_top_n = 0

    run_evolution(config)


def _run_batch_backtest(
    *,
    panel_data_path: str,
    text_data_path: str,
    write_data_artifacts: bool,
    debug_symbol_count: int,
    debug_time_steps: int,
    log_base: Path,
    min_factors: int = 10,
    ic_threshold: float = 0.003,
    icir_threshold: float = 0.02,
) -> None:
    """调用 scripts/run_batch_backtest.py 的批量回测流程。"""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from scripts.run_batch_backtest import run_batch_backtest

    print(f"Running batch backtest, logs: {log_base}")
    effective_panel_path, _bundle = _prepare_data_bundle(
        panel_data_path=panel_data_path,
        text_data_path=text_data_path,
        log_base=log_base,
        write_data_artifacts=write_data_artifacts,
        debug_symbol_count=debug_symbol_count,
        debug_time_steps=debug_time_steps,
    )

    result = run_batch_backtest(
        panel_data_path=Path(effective_panel_path),
        min_factors=min_factors,
        output_dir=log_base,
        ic_threshold=ic_threshold,
        icir_threshold=icir_threshold,
    )
    print(f"Batch backtest result: {result['status']}")

    # 自动生成 workflow_output 分析图表
    if result.get("status") == "completed":
        run_out = Path(result.get("_output_dir", str(log_base)))
        try:
            import importlib.util
            _script_path = PROJECT_ROOT / "scripts" / "generate_workflow_analysis.py"
            _spec = importlib.util.spec_from_file_location("generate_workflow_analysis", _script_path)
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            panel_path = Path(effective_panel_path)
            _mod.run_analysis(run_out, panel_path, output_dir=run_out / "figures")
            print(f"Workflow analysis charts generated: {run_out / 'figures'}")
        except Exception as e:
            print(f"Workflow analysis failed: {e}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-agent-core alpha factor mining workflow.")
    parser.add_argument("--mode", choices=["mining", "evolution", "batch-backtest"], default="mining",
                        help="Run mode: mining (default), evolution, or batch-backtest.")
    parser.add_argument("--panel-data-path", default=str(PROJECT_ROOT / "data" / "panel_data.parquet"))
    parser.add_argument("--text-data-path", default="",
                        help="Optional local JSONL/CSV market text data to merge into the panel.")
    parser.add_argument("--debug-symbol-count", type=int, default=20,
                        help="Maximum symbols in generated debug panel.")
    parser.add_argument("--debug-time-steps", type=int, default=180,
                        help="Maximum time steps in generated debug panel.")
    parser.add_argument("--write-data-artifacts", action="store_true", default=False,
                        help="Write data_bundle artifacts even when no text data is provided.")
    parser.add_argument("--loop-count", type=int, default=1)
    parser.add_argument("--initial-direction", default="", help="Optional initial hypothesis direction.")
    parser.add_argument("--initial-hypothesis", default="", help="Optional seed hypothesis injected into first loop.")
    parser.add_argument("--stop-on-error", action="store_true", default=False)
    parser.add_argument("--retry-per-round", type=int, default=0)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--evo-generations", type=int, default=5,
                        help="Evolution mode: number of generations (default: 5)")
    parser.add_argument("--evo-population", type=int, default=10,
                        help="Evolution mode: population size (default: 10)")
    parser.add_argument("--evolve-target", choices=["factor", "model_params"], default="factor",
                        help="Evolution mode: optimize factors (default) or model parameters.")
    parser.add_argument("--factor-ga-mode", choices=["hybrid", "expression", "subset"], default="hybrid",
                        help="Evolution factor target: expression, subset, or hybrid.")
    parser.add_argument("--evo-seed", type=int, default=42,
                        help="Evolution mode: random seed (default: 42)")
    parser.add_argument("--min-factors", type=int, default=10,
                        help="Batch backtest: minimum factors required (default: 10)")
    parser.add_argument("--ic-threshold", type=float, default=0.003,
                        help="Batch backtest: minimum absolute IC mean threshold (default: 0.003)")
    parser.add_argument("--icir-threshold", type=float, default=0.02,
                        help="Batch backtest: minimum absolute ICIR threshold (default: 0.02)")
    return parser


def main() -> None:
    _load_dotenv_files()
    parser = build_arg_parser()
    args = parser.parse_args()
    debug_enabled = bool(args.debug)
    debug_log_path = _build_debug_log_path(debug_enabled)

    # 创建结构化日志文件（logs/alpha_factor_mining_loop/<run_id>/）
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_base = PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / run_id
    log_base.mkdir(parents=True, exist_ok=True)

    # --- Always-on JSONL loggers in the run log directory ---
    progress_jsonl_logger = _build_jsonl_logger(log_base / "progress.jsonl", "workflow_event")
    llm_raw_io_jsonl_logger = _build_jsonl_logger(log_base / "llm_raw_io.jsonl", "llm_io")

    # --- Optional verbose debug loggers (stderr + consolidated debug file) ---
    progress_debug_logger = _build_debug_logger(debug_enabled, "workflow_event", debug_log_path)
    raw_io_debug_logger = _build_debug_logger(debug_enabled, "llm_io", debug_log_path)

    progress_logger = _chain_loggers(progress_jsonl_logger, progress_debug_logger)
    raw_io_logger = _chain_loggers(llm_raw_io_jsonl_logger, raw_io_debug_logger)

    if args.mode == "evolution":
        _run_evolution(
            panel_data_path=args.panel_data_path,
            text_data_path=args.text_data_path,
            write_data_artifacts=args.write_data_artifacts,
            debug_symbol_count=args.debug_symbol_count,
            debug_time_steps=args.debug_time_steps,
            debug_enabled=debug_enabled,
            log_base=log_base,
            progress_logger=progress_logger,
            raw_io_logger=raw_io_logger,
            evolve_target=args.evolve_target,
            factor_ga_mode=args.factor_ga_mode,
            num_generations=args.evo_generations,
            population_size=args.evo_population,
            seed=args.evo_seed,
        )
        return

    if args.mode == "batch-backtest":
        _run_batch_backtest(
            panel_data_path=args.panel_data_path,
            text_data_path=args.text_data_path,
            write_data_artifacts=args.write_data_artifacts,
            debug_symbol_count=args.debug_symbol_count,
            debug_time_steps=args.debug_time_steps,
            log_base=log_base,
            min_factors=args.min_factors,
            ic_threshold=args.ic_threshold,
            icir_threshold=args.icir_threshold,
        )
        return

    # --- mining mode (default): subprocess to run_alpha_factor_mining_loop.py ---
    script = PROJECT_ROOT / "scripts" / "run_alpha_factor_mining_loop.py"
    cmd = [
        sys.executable, str(script),
        "--run-id", run_id,
        "--panel-data-path", args.panel_data_path,
        "--loop-count", str(args.loop_count),
        "--initial-direction", args.initial_direction,
        "--initial-hypothesis", args.initial_hypothesis,
        "--retry-per-round", str(args.retry_per_round),
    ]
    if args.stop_on_error:
        cmd.append("--stop-on-error")
    if args.text_data_path:
        cmd.extend(["--text-data-path", args.text_data_path])
    if args.write_data_artifacts:
        cmd.append("--write-data-artifacts")
    cmd.extend(["--debug-symbol-count", str(args.debug_symbol_count)])
    cmd.extend(["--debug-time-steps", str(args.debug_time_steps)])
    print(f"Running mining mode: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
