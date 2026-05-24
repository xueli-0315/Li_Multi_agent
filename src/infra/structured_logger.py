"""structured_logger.py
~~~~~~~~~~~~~~~~~~~~~~~
统一结构化日志模块，替代各入口文件中分散的临时日志函数。

每条日志输出为 JSON 行，包含标准化字段：
  timestamp, run_id, level, category, agent_name, loop_index, phase, message, payload

日志可被 AI 辅助工具或 jq 按字段过滤：
  jq 'select(.agent_name=="factor_coder_agent")' run.log
  jq 'select(.category=="factor" and .payload.acceptable==false)' run.log

同时向 stderr 输出人类可读的摘要行，方便实时监控。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


_LEVEL_SYMBOLS = {
    "INFO": "✅",
    "WARN": "⚠️",
    "ERROR": "❌",
    "DEBUG": "🔍",
}


class StructuredLogger:
    """统一结构化日志输出器。

    Args:
        run_id: 本次运行的唯一标识（如时间戳字符串）
        log_file: 日志写入的 JSONL 文件路径（None 表示仅输出到 stderr）
        verbose: 为 True 时在 stderr 额外打印 payload 摘要
        min_level: 最低输出级别（DEBUG/INFO/WARN/ERROR）
    """

    _LEVELS = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3}

    def __init__(
        self,
        run_id: str = "",
        log_file: str | Path | None = None,
        *,
        verbose: bool = False,
        min_level: str = "INFO",
    ) -> None:
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = Path(log_file) if log_file else None
        self.verbose = verbose
        self.min_level_val = self._LEVELS.get(min_level.upper(), 1)
        self._file_handle = None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._file_handle = self.log_file.open("a", encoding="utf-8")

    def __del__(self) -> None:
        if self._file_handle is not None:
            try:
                self._file_handle.close()
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────
    # Core emit
    # ─────────────────────────────────────────────────────────────

    def log_event(
        self,
        *,
        level: str = "INFO",
        category: str = "workflow",
        agent_name: str = "",
        loop_index: int = 0,
        phase: str = "",
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """输出一条结构化日志事件。"""
        if self._LEVELS.get(level.upper(), 1) < self.min_level_val:
            return

        event: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "level": level.upper(),
            "category": category,
            "agent_name": agent_name,
            "loop_index": loop_index,
            "phase": phase or "—",
            "message": message,
            "payload": payload or {},
        }

        # 写入 JSONL 文件
        if self._file_handle is not None:
            try:
                self._file_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                self._file_handle.flush()
            except Exception:
                pass

        # 输出人类可读行到 stderr
        self._print_human_line(event)

    def _print_human_line(self, event: dict[str, Any]) -> None:
        symbol = _LEVEL_SYMBOLS.get(event["level"], "•")
        ts = event["timestamp"][11:23]  # HH:MM:SS.mmm
        loop = f"Loop{event['loop_index']:02d}" if event["loop_index"] else "     "
        agent = f"[{event['agent_name']}]" if event["agent_name"] else ""
        phase = f"({event['phase']})" if event.get("phase") and event["phase"] != "—" else ""
        line = f"{symbol} {ts} {loop} {event['category']:10s} {agent}{phase} {event['message']}"
        print(line, file=sys.stderr)
        if self.verbose and event["payload"]:
            # 打印 payload 的关键字段（最多3个）
            keys = list(event["payload"].keys())[:3]
            extras = " | ".join(f"{k}={str(event['payload'][k])[:40]}" for k in keys)
            print(f"   └─ {extras}", file=sys.stderr)

    # ─────────────────────────────────────────────────────────────
    # Convenience shortcuts
    # ─────────────────────────────────────────────────────────────

    def info(self, message: str, *, agent_name: str = "", loop_index: int = 0,
             phase: str = "", category: str = "workflow",
             payload: dict[str, Any] | None = None) -> None:
        self.log_event(level="INFO", category=category, agent_name=agent_name,
                       loop_index=loop_index, phase=phase, message=message, payload=payload)

    def warn(self, message: str, *, agent_name: str = "", loop_index: int = 0,
             phase: str = "", category: str = "workflow",
             payload: dict[str, Any] | None = None) -> None:
        self.log_event(level="WARN", category=category, agent_name=agent_name,
                       loop_index=loop_index, phase=phase, message=message, payload=payload)

    def error(self, message: str, *, agent_name: str = "", loop_index: int = 0,
              phase: str = "", category: str = "workflow",
              payload: dict[str, Any] | None = None) -> None:
        self.log_event(level="ERROR", category=category, agent_name=agent_name,
                       loop_index=loop_index, phase=phase, message=message, payload=payload)

    def debug(self, message: str, *, agent_name: str = "", loop_index: int = 0,
              phase: str = "", category: str = "workflow",
              payload: dict[str, Any] | None = None) -> None:
        self.log_event(level="DEBUG", category=category, agent_name=agent_name,
                       loop_index=loop_index, phase=phase, message=message, payload=payload)

    # ─────────────────────────────────────────────────────────────
    # Domain-specific helpers（与现有日志锚点对应）
    # ─────────────────────────────────────────────────────────────

    def log_agent_start(self, agent_name: str, *, loop_index: int = 0,
                        phase: str = "", input_keys: list[str] | None = None) -> None:
        self.info(
            f"Agent started: {agent_name}",
            agent_name=agent_name, loop_index=loop_index, phase=phase,
            category="agent",
            payload={"input_keys": input_keys or []},
        )

    def log_agent_finish(self, agent_name: str, *, loop_index: int = 0,
                         phase: str = "", duration_ms: int = 0,
                         output_keys: list[str] | None = None,
                         error: str | None = None) -> None:
        level = "ERROR" if error else "INFO"
        self.log_event(
            level=level, category="agent",
            agent_name=agent_name, loop_index=loop_index, phase=phase,
            message=f"Agent {'failed' if error else 'finished'}: {agent_name}",
            payload={"duration_ms": duration_ms, "output_keys": output_keys or [], "error": error},
        )

    def log_factor_eval(
        self,
        *,
        agent_name: str,
        loop_index: int = 0,
        phase: str = "",
        factor_name: str,
        expression: str,
        parsable: bool,
        acceptable: bool,
        execution_ok: bool,
        non_na_count: int = 0,
        duration_ms: int = 0,
    ) -> None:
        level = "INFO" if acceptable else "WARN"
        self.log_event(
            level=level, category="factor",
            agent_name=agent_name, loop_index=loop_index, phase=phase,
            message=f"Factor eval: {factor_name} | accept={acceptable}",
            payload={
                "factor_name": factor_name,
                "expression": expression[:120],
                "parsable": parsable,
                "acceptable": acceptable,
                "execution_ok": execution_ok,
                "non_na_count": non_na_count,
                "duration_ms": duration_ms,
            },
        )

    def log_quality_gate(
        self,
        *,
        loop_index: int = 0,
        accepted: bool,
        reason: str,
        gate_name: str,
        metrics_summary: dict[str, Any] | None = None,
    ) -> None:
        level = "INFO" if accepted else "WARN"
        self.log_event(
            level=level, category="factor",
            agent_name="factor_quality_gate", loop_index=loop_index,
            message=f"Quality gate [{gate_name}]: {'ACCEPT' if accepted else 'REJECT'} — {reason}",
            payload={"accepted": accepted, "gate_name": gate_name,
                     "reason": reason, "metrics": metrics_summary or {}},
        )

    def log_loop_summary(
        self,
        *,
        loop_index: int,
        loop_ok: bool,
        accepted_factors: list[str],
        failed_factors: list[str],
        duration_s: float,
        ic: float | None = None,
        turnover: float | None = None,
    ) -> None:
        level = "INFO" if loop_ok else "WARN"
        self.log_event(
            level=level, category="workflow",
            agent_name="", loop_index=loop_index,
            message=f"Loop {loop_index:02d} {'OK' if loop_ok else 'FAILED'} | "
                    f"accepted={len(accepted_factors)}, failed={len(failed_factors)}, "
                    f"time={duration_s:.1f}s",
            payload={
                "loop_ok": loop_ok,
                "accepted_factors": accepted_factors,
                "failed_factors": failed_factors,
                "duration_s": duration_s,
                "IC": ic,
                "turnover": turnover,
            },
        )

    def log_context_compression(
        self,
        *,
        agent_name: str,
        loop_index: int = 0,
        original_size: int,
        compressed_size: int,
    ) -> None:
        ratio = compressed_size / max(1, original_size)
        self.debug(
            f"Context compressed: {original_size} → {compressed_size} chars ({ratio:.0%})",
            agent_name=agent_name, loop_index=loop_index,
            category="workflow",
            payload={"original_size": original_size,
                     "compressed_size": compressed_size,
                     "compression_ratio": round(ratio, 3)},
        )
