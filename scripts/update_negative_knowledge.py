import os
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs" / "alpha_factor_mining_loop"
KNOWLEDGE_DIR = PROJECT_ROOT / "factor_library" / "raw" / "negative_knowledge"
FAILURES_FILE = KNOWLEDGE_DIR / "all_failures.jsonl"

def robust_sync():
    """
    终极召回逻辑：
    1. 遍历所有 logs 下的 .json 和 .jsonl
    2. 提取所有提议的因子名、表达式
    3. 提取所有结构化日志中的指标和报错
    4. 不去重，按 RunID 和时间排序
    """
    all_entries = {}
    
    # 首先加载现有的（如果有的话，防止彻底丢失）
    if FAILURES_FILE.exists():
        with open(FAILURES_FILE, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    key = f"{d.get('name')}_{d.get('run_id')}"
                    all_entries[key] = d
                except: continue

    print(f"Initial record count: {len(all_entries)}")

    # 递归遍历所有日志
    for root, dirs, files in os.walk(LOGS_DIR):
        run_id = Path(root).name
        for file in files:
            file_path = Path(root) / file
            
            # A. 处理 loop_*.json
            if file.startswith("loop_") and file.endswith(".json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        # 1. 提取提议因子
                        for pf in data.get("proposed_factors", []):
                            f_name = pf.get("name")
                            if f_name:
                                key = f"{f_name}_{run_id}"
                                if key not in all_entries:
                                    all_entries[key] = {
                                        "name": f_name,
                                        "run_id": run_id,
                                        "expression": pf.get("expression", "N/A"),
                                        "timestamp": data.get("timestamp", ""),
                                        "reason": "Proposed but results pending or rejected.",
                                        "type": "unsuccessful_attempt"
                                    }
                        
                        # 2. 提取 shared_context 里的被拒绝因子 (历史累积)
                        rejected = data.get("shared_context", {}).get("rejected_factors", [])
                        for rf in rejected:
                            f_name = rf.get("name")
                            if f_name:
                                key = f"{f_name}_{run_id}"
                                if key not in all_entries:
                                    all_entries[key] = {"name": f_name, "run_id": run_id}
                                all_entries[key].update({
                                    "reason": rf.get("reason", ""),
                                    "metrics": rf.get("metrics", {}),
                                    "timestamp": rf.get("timestamp", ""),
                                    "type": "low_metric"
                                })
                except: continue

            # B. 处理 structured.jsonl
            elif file == "structured.jsonl":
                try:
                    with open(file_path, "r") as f:
                        for line in f:
                            event = json.loads(line)
                            payload = event.get("payload", {})
                            msg = event.get("message", "")
                            
                            # 识别回测失败
                            if event.get("category") == "factor" and "REJECTED" in msg:
                                f_name = payload.get("factor_name")
                                if f_name:
                                    key = f"{f_name}_{run_id}"
                                    if key not in all_entries: all_entries[key] = {"name": f_name, "run_id": run_id}
                                    all_entries[key].update({
                                        "reason": payload.get("reason"),
                                        "metrics": payload.get("metrics"),
                                        "timestamp": event.get("timestamp"),
                                        "type": "low_metric"
                                    })
                            
                            # 识别系统报错
                            elif event.get("level") == "ERROR":
                                err_msg = event.get("message") or ""
                                f_name = payload.get("factor_name") or payload.get("name")
                                if not f_name:
                                    m = re.search(r"factor\s+([A-Za-z0-9_]+)", err_msg)
                                    if m: f_name = m.group(1)
                                
                                if f_name:
                                    key = f"{f_name}_{run_id}"
                                    if key not in all_entries: all_entries[key] = {"name": f_name, "run_id": run_id}
                                    all_entries[key].update({
                                        "reason": f"System Error: {err_msg}",
                                        "timestamp": event.get("timestamp"),
                                        "type": "error"
                                    })
                except: continue

    # 保存
    with open(FAILURES_FILE, "w") as f:
        # 按时间排序导出，保证 jsonl 逻辑顺序
        sorted_keys = sorted(all_entries.keys(), key=lambda k: all_entries[k].get("timestamp", ""))
        for k in sorted_keys:
            f.write(json.dumps(all_entries[k]) + "\n")
            
    print(f"Final Count: {len(all_entries)} entries successfully synchronized.")

if __name__ == "__main__":
    robust_sync()
