from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llm import GatewayModelClient, LLMGateway, RetryPolicy
from llm.providers import (
    AzureOpenAIProvider,
    OpenAICompatibleProvider,
    ZhipuNativeProvider,
)
from factor_runtime import FactorLibraryManager
from infra import StructuredLogger
from core import reset_current_loop_index, set_current_loop_index
from adapters import CrossSectionDomainAdapter
from workflows import AlphaFactorMiningWorkflow
from workflows.trajectory_pool_helper import TrajectoryPool


def _parse_env_line(raw_line: str) -> tuple[str, str] | None:
    """解析 .env 单行文本，返回 (key, value)；注释、空行、非法行返回 None。"""
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
    """按顺序加载 .env 与 .env.local，并且不覆盖已有系统环境变量。"""
    env_candidates = [PROJECT_ROOT / ".env", PROJECT_ROOT / ".env.local"]
    for env_file in env_candidates:
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
    """把 Path、datetime、tuple 等对象递归转换为可 JSON 序列化的结构。"""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _require_env(keys: list[str]) -> dict[str, str]:
    """批量读取必需环境变量，缺失时抛出带字段名的错误。"""
    values: dict[str, str] = {}
    missing: list[str] = []
    for key in keys:
        raw = os.getenv(key, "").strip()
        if not raw:
            missing.append(key)
        else:
            values[key] = raw
    if missing:
        missing_text = ", ".join(missing)
        raise RuntimeError(f"缺少环境变量: {missing_text}")
    return values


def _build_real_model_client(
    raw_io_logger: Callable[[dict[str, Any]], None] | None = None,
) -> GatewayModelClient:
    """根据 LLM_PROVIDER 与相关环境变量构建真实可用的模型客户端。"""
    provider_name = os.getenv("LLM_PROVIDER", "zhipu").strip().lower()
    default_model = os.getenv("LLM_MODEL", "").strip() or None
    raw_agent_model_map = os.getenv("AGENT_MODEL_MAP", "").strip()
    agent_model_map: dict[str, str] = {}
    if raw_agent_model_map:
        parsed_agent_model_map = json.loads(raw_agent_model_map)
        if not isinstance(parsed_agent_model_map, dict):
            raise RuntimeError("AGENT_MODEL_MAP 必须是 JSON 对象")
        agent_model_map = {
            str(agent_name): str(model_name)
            for agent_name, model_name in parsed_agent_model_map.items()
            if str(model_name).strip()
        }
    if provider_name == "zhipu":
        envs = _require_env(["ZHIPU_API_KEY"])
        model_name = os.getenv("ZHIPU_MODEL", "").strip() or default_model or "glm-4-flash"
        provider = ZhipuNativeProvider(
            api_key=envs["ZHIPU_API_KEY"],
        )
        gateway = LLMGateway(
            provider=provider,
            retry_policy=RetryPolicy(max_retries=2, wait_seconds=1.0),
            default_model=model_name,
            raw_io_logger=raw_io_logger,
        )
        return GatewayModelClient(gateway, agent_model_overrides=agent_model_map)
    if provider_name == "azure_openai":
        envs = _require_env(
            [
                "AZURE_OPENAI_API_KEY",
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_API_VERSION",
            ]
        )
        model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip() or default_model
        if not model_name:
            raise RuntimeError("缺少环境变量: AZURE_OPENAI_DEPLOYMENT 或 LLM_MODEL")
        provider = AzureOpenAIProvider(
            api_key=envs["AZURE_OPENAI_API_KEY"],
            azure_endpoint=envs["AZURE_OPENAI_ENDPOINT"],
            api_version=envs["AZURE_OPENAI_API_VERSION"],
        )
        gateway = LLMGateway(
            provider=provider,
            retry_policy=RetryPolicy(max_retries=2, wait_seconds=1.0),
            default_model=model_name,
            raw_io_logger=raw_io_logger,
        )
        return GatewayModelClient(gateway, agent_model_overrides=agent_model_map)
    envs = _require_env(["OPENAI_API_KEY", "LLM_MODEL"])
    base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
    provider = OpenAICompatibleProvider(
        api_key=envs["OPENAI_API_KEY"],
        base_url=base_url,
    )
    gateway = LLMGateway(
        provider=provider,
        retry_policy=RetryPolicy(max_retries=2, wait_seconds=1.0),
        default_model=envs["LLM_MODEL"],
        raw_io_logger=raw_io_logger,
    )
    return GatewayModelClient(gateway, agent_model_overrides=agent_model_map)


def _save_trace_log(loop_trace: Any, output_dir: Path, loop_index: int) -> None:
    """把单轮 loop 的完整执行轨迹写入 loop_{index}.json。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    loop_payload = {
        "loop_index": loop_index,
        "record_count": len(loop_trace.records),
        "records": [
            {
                "agent_name": record.agent_name,
                "started_at": record.started_at.isoformat(),
                "ended_at": record.ended_at.isoformat(),
                "input_snapshot": _json_safe(record.input_snapshot),
                "output_snapshot": _json_safe(record.output_snapshot),
                "error": record.error,
            }
            for record in loop_trace.records
        ],
        "final_shared_context": _json_safe(loop_trace.final_shared_context),
    }
    output_path = output_dir / f"loop_{loop_index:03d}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(loop_payload, file, ensure_ascii=False, indent=2)


def _persist_factor_library(
    loop_trace: Any,
    output_file: Path,
    panel_data_path: Path,
    loop_index: int,
) -> None:
    """以前这里会手动追加因子逻辑，现在因子库写入完全由 workflow 内的质量门控负责，
    以确保只有达标的因子会被记录。此函数现在仅作为扩展预留。"""
    pass


def _extract_keys(payload: Any) -> list[str]:
    """从字典中提取并排序键名，用于进度日志的轻量摘要。"""
    if not isinstance(payload, dict):
        return []
    return sorted(str(key) for key in payload.keys())


def _append_progress(progress_file: Path, payload: dict[str, Any]) -> None:
    """把进度事件追加到 progress.jsonl，同时输出到 stderr 做实时可视化。"""
    progress_file.parent.mkdir(parents=True, exist_ok=True)
    with progress_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_safe(payload), ensure_ascii=False))
        file.write("\n")
    print(json.dumps(_json_safe(payload), ensure_ascii=False), file=sys.stderr, flush=True)


def _append_raw_llm_io(raw_io_file: Path, payload: dict[str, Any]) -> None:
    raw_io_file.parent.mkdir(parents=True, exist_ok=True)
    with raw_io_file.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, indent=2))
        file.write("\n\n")


def main() -> None:
    """运行多轮 Alpha 因子挖掘工作流，并产出进度日志、轨迹日志和最终汇总。"""
    import argparse
    parser = argparse.ArgumentParser(description="Run multi-agent alpha factor mining loop.")
    parser.add_argument("--run-id", default="", help="Run ID from main.py (shared log directory)")
    parser.add_argument("--panel-data-path", default=None,
                        help="Path to panel data parquet (default: data/panel_data.parquet)")
    parser.add_argument("--text-data-path", default="",
                        help="Optional local JSONL/CSV market text data to merge into the panel")
    parser.add_argument("--market-type", choices=["crypto", "stock", "futures"], default="crypto")
    parser.add_argument("--domain-config", default="", help="Optional domain adapter YAML config path.")
    parser.add_argument("--symbol-alias-path", default="", help="Optional symbol alias YAML/JSON path.")
    parser.add_argument("--debug-symbol-count", type=int, default=20,
                        help="Maximum symbols in generated debug panel")
    parser.add_argument("--debug-time-steps", type=int, default=180,
                        help="Maximum time steps in generated debug panel")
    parser.add_argument("--write-data-artifacts", action="store_true", default=False,
                        help="Write data_bundle artifacts even when no text data is provided")
    parser.add_argument("--loop-count", type=int, default=15)
    parser.add_argument("--initial-direction", default="", help="Initial hypothesis direction")
    parser.add_argument("--initial-hypothesis", default="", help="Seed hypothesis")
    parser.add_argument("--stop-on-error", action="store_true", default=False)
    parser.add_argument("--retry-per-round", type=int, default=0)
    args = parser.parse_args()

    _load_dotenv_files()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "logs" / "alpha_factor_mining_loop" / run_id
    raw_llm_io_file = log_dir / "llm_raw_io.jsonl"
    # --- Step-5: 实例化结构化日志模块 ---
    logger = StructuredLogger(
        run_id=run_id,
        log_file=log_dir / "structured.jsonl",
        verbose=True,
    )
    model_client = _build_real_model_client(
        raw_io_logger=lambda payload: _append_raw_llm_io(raw_llm_io_file, payload)
    )
    panel_data_path = Path(args.panel_data_path) if args.panel_data_path else PROJECT_ROOT / "data" / "panel_data.parquet"
    adapter = CrossSectionDomainAdapter(
        panel_data_path,
        market_type=args.market_type,
        text_data_path=args.text_data_path or None,
        artifact_dir=log_dir / "data_bundle",
        write_artifacts=args.write_data_artifacts or bool(args.text_data_path),
        debug_symbol_count=args.debug_symbol_count,
        debug_time_steps=args.debug_time_steps,
        domain_config_path=args.domain_config or None,
        symbol_alias_path=args.symbol_alias_path or None,
    )
    shared_payload = adapter.build_initial_payload()
    default_initial_direction = "山寨币在过热时可能会发生反转，基于这个假设可以发掘盈利因子。"
    initial_direction_raw = args.initial_direction or os.getenv("INITIAL_DIRECTION", default_initial_direction)
    initial_direction = initial_direction_raw.strip() or None
    traces: list[Any] = []
    loop_count = args.loop_count
    factor_library_file = PROJECT_ROOT / "factor_library" / "factor_library.jsonl"
    progress_file = log_dir / "progress.jsonl"
    progress_state: dict[str, int] = {"loop_index": 0}
    trajectory_pool = TrajectoryPool()

    def on_progress(event: dict[str, object]) -> None:
        stage = str(event.get("stage", ""))
        agent_name = str(event.get("agent_name", ""))
        loop_idx = int(progress_state["loop_index"])
        base = {
            "timestamp": datetime.now().isoformat(),
            "run_id": run_id,
            "loop_index": loop_idx,
            "agent_name": agent_name,
            "stage": stage,
        }
        if stage == "agent_started":
            input_snapshot = event.get("input_snapshot", {})
            private_context = input_snapshot.get("private_context", {}) if isinstance(input_snapshot, dict) else {}
            shared_context = input_snapshot.get("shared_context", {}) if isinstance(input_snapshot, dict) else {}
            base["input_private_keys"] = _extract_keys(private_context)
            base["input_shared_keys"] = _extract_keys(shared_context)
            logger.log_agent_start(
                agent_name, loop_index=loop_idx,
                input_keys=_extract_keys(shared_context),
            )
        elif stage == "agent_finished":
            output_snapshot = event.get("output_snapshot", {})
            shared_updates = output_snapshot.get("shared_updates", {}) if isinstance(output_snapshot, dict) else {}
            artifacts = output_snapshot.get("artifacts", {}) if isinstance(output_snapshot, dict) else {}
            base["output_shared_update_keys"] = _extract_keys(shared_updates)
            base["output_artifact_keys"] = _extract_keys(artifacts)
            logger.log_agent_finish(
                agent_name, loop_index=loop_idx,
                output_keys=_extract_keys(shared_updates),
            )
        elif stage == "agent_failed":
            output_snapshot = event.get("output_snapshot", {})
            shared_context = output_snapshot.get("shared_context", {}) if isinstance(output_snapshot, dict) else {}
            base["error"] = str(event.get("error", ""))
            base["output_shared_keys"] = _extract_keys(shared_context)
            logger.log_agent_finish(
                agent_name, loop_index=loop_idx,
                error=base["error"],
            )
        _append_progress(progress_file, base)

    workflow = AlphaFactorMiningWorkflow(
        model_client=model_client,
        progress_callback=on_progress,
        logger=logger,
    )
    for loop_index in range(1, loop_count + 1):
        progress_state["loop_index"] = loop_index
        loop_started_at = datetime.now()
        logger.info(
            f"Loop {loop_index:02d} started",
            loop_index=loop_index, category="workflow",
        )
        _append_progress(
            progress_file,
            {
                "timestamp": loop_started_at.isoformat(),
                "run_id": run_id,
                "loop_index": loop_index,
                "stage": "loop_started",
            },
        )
        loop_token = set_current_loop_index(loop_index)
        try:
            trace = workflow.run(
                shared_payload,
                initial_direction=initial_direction if loop_index == 1 else None,
                initial_hypothesis=shared_payload.get("initial_hypothesis") if loop_index == 1 else None,
                loop_index=loop_index,
            )
        except Exception as e:
            if args.stop_on_error:
                raise
            logger.info(f"Loop {loop_index} failed (continuing): {e}", loop_index=loop_index, category="workflow")
            # Create empty trace for error tracking
            from core import LoopTrace
            trace = LoopTrace(loop_index=loop_index, records=[], final_shared_context=shared_payload)
            trace.records  # force-init
        finally:
            reset_current_loop_index(loop_token)

        # retry logic
        retries_used = 0
        while args.retry_per_round > 0 and retries_used < args.retry_per_round and trace.records and any(r.error is not None for r in trace.records):
            retries_used += 1
            logger.info(f"Loop {loop_index} retry {retries_used}/{args.retry_per_round}", loop_index=loop_index, category="workflow")
            loop_token = set_current_loop_index(loop_index)
            try:
                trace = workflow.run(
                    shared_payload,
                    initial_direction=None,
                    loop_index=loop_index,
                )
            except Exception:
                if args.stop_on_error:
                    raise
            finally:
                reset_current_loop_index(loop_token)
        traces.append(trace)
        _save_trace_log(trace, log_dir, loop_index)
        _persist_factor_library(trace, factor_library_file, panel_data_path, loop_index)
        final_shared_payload = dict(trace.final_shared_context or shared_payload)
        trajectory_pool.add(final_shared_payload, loop_index)
        shared_payload = final_shared_payload
        error_count = len([r for r in trace.records if r.error is not None])
        loop_duration_s = (datetime.now() - loop_started_at).total_seconds()
        final_metrics = shared_payload.get("metrics", {})
        
        # 优先使用质量门控的真实结论
        impl_meta = shared_payload.get("factor_implementation", {})
        gate_ok = impl_meta.get("gate_accepted", False) if isinstance(impl_meta, dict) else False

        ic_val = final_metrics.get("IC", final_metrics.get("Rank IC")) if isinstance(final_metrics, dict) else None

        logger.log_loop_summary(
            loop_index=loop_index,
            loop_ok=gate_ok,  # 这里反映真实的门控结果
            accepted_factors=list(impl_meta.get("accepted_factors", [])) if gate_ok else [],
            failed_factors=list(impl_meta.get("failed_factors", [])),
            duration_s=loop_duration_s,
            ic=float(ic_val) if ic_val is not None else None,
            turnover=float(final_metrics["turnover"]) if isinstance(final_metrics, dict) and "turnover" in final_metrics else None,
        )
        _append_progress(
            progress_file,
            {
                "timestamp": datetime.now().isoformat(),
                "run_id": run_id,
                "loop_index": loop_index,
                "stage": "loop_finished",
                "agent_count": len(trace.records),
                "error_count": error_count,
            }
        )
        
        # --- [NEW] 1. 更新负面知识库 ---
        print(f"\n[Round {loop_index}] 步骤A: 更新负面知识库 (Raw Data)...")
        import subprocess
        try:
            update_script = PROJECT_ROOT / "scripts" / "update_negative_knowledge.py"
            subprocess.run(["python3", str(update_script)], check=True)
        except Exception as e:
            print(f"负面知识库更新失败: {e}")

        # --- [NEW] 2. 执行知识蒸馏 (每隔 5 轮) ---
        if loop_index % 5 == 0:
            print(f"[Round {loop_index}] 步骤B: 执行错题总结 (Knowledge Distillation)...")
            try:
                distill_script = PROJECT_ROOT / "scripts" / "distill_negative_knowledge.py"
                subprocess.run(["python3", str(distill_script)], check=True)
            except Exception as e:
                print(f"知识蒸馏失败: {e}")

        # --- [NEW] 3. 同步 Wiki 门户 (成功 + 失败) ---
        print(f"[Round {loop_index}] 步骤C: 同步 Factor Library Wiki (Portal)...")
        try:
            wiki_script = PROJECT_ROOT / "scripts" / "build_factor_wiki.py"
            neg_wiki_script = PROJECT_ROOT / "scripts" / "build_negative_wiki.py"
            subprocess.run(["python3", str(wiki_script)], check=True)
            subprocess.run(["python3", str(neg_wiki_script)], check=True)
        except Exception as e:
            print(f"Wiki 实时同步失败: {e}")

    trace = traces[-1]
    output = {
        "run_id": run_id,
        "loop_count": len(traces),
        "record_count_per_loop": [len(loop_trace.records) for loop_trace in traces],
        "agents_per_loop": [[record.agent_name for record in loop_trace.records] for loop_trace in traces],
        "final_shared_context_keys": sorted(list(trace.final_shared_context.keys())),
        "final_shared_context": trace.final_shared_context,
        "initial_direction": initial_direction,
        "log_dir": str(log_dir),
        "raw_llm_io_file": str(raw_llm_io_file),
        "factor_library_file": str(factor_library_file),
        "errors": [
            [record.error for record in loop_trace.records if record.error is not None]
            for loop_trace in traces
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    
    # --- 最终全量同步 Wiki 知识库 (兜底) ---
    print("\n" + "="*50)
    print("正在执行最终 Wiki 同步...")
    try:
        wiki_script = PROJECT_ROOT / "scripts" / "build_factor_wiki.py"
        subprocess.run(["python3", str(wiki_script)], check=True)
        print("Wiki 最终同步成功！")
    except Exception as e:
        print(f"Wiki 最终同步失败: {e}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
