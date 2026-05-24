import json
import os
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FAILURES_FILE = PROJECT_ROOT / "factor_library" / "raw" / "negative_knowledge" / "all_failures.jsonl"
WIKI_DIR = PROJECT_ROOT / "factor_library" / "wiki"
FAILURES_WIKI_DIR = WIKI_DIR / "failures"

def generate_failure_wiki():
    """将 all_failures.jsonl 中的每个记录转换为 Wiki 页面"""
    if not FAILURES_FILE.exists():
        print("No failures file found.")
        return

    FAILURES_WIKI_DIR.mkdir(parents=True, exist_ok=True)
    
    failures = []
    with open(FAILURES_FILE, "r") as f:
        for line in f:
            try:
                failures.append(json.loads(line))
            except: continue

    # 按时间倒序排列
    failures.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    for i, fail in enumerate(failures):
        name = fail.get('name', 'Unknown_Factor').replace("/", "_")
        run_id = fail.get('run_id', 'unknown')
        timestamp = fail.get('timestamp', 'unknown')
        fail_type = fail.get('type', 'unknown')
        
        # 文件名：日期_RunID_索引_名称.md
        safe_name = f"{timestamp[:10].replace('-', '')}_{run_id}_{i}_{name}.md"
        file_path = FAILURES_WIKI_DIR / safe_name
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# Failure Report: {name}\n\n")
            f.write(f"## Metadata\n")
            f.write(f"- **Timestamp:** {timestamp}\n")
            f.write(f"- **Run ID:** `{run_id}`\n")
            f.write(f"- **Failure Type:** `{fail_type}`\n\n")
            
            f.write(f"## Reason for Rejection\n")
            f.write(f"> {fail.get('reason', 'No reason provided.')}\n\n")
            
            if fail.get('metrics'):
                f.write(f"## Performance Metrics\n")
                f.write("```json\n")
                f.write(json.dumps(fail['metrics'], indent=4))
                f.write("\n```\n\n")
            
            if fail.get('expression'):
                f.write(f"## Factor Expression\n")
                f.write(f"```python\n{fail['expression']}\n```\n\n")

    # 更新索引文件 (Negative Index)
    update_index(failures)

def update_index(failures):
    index_path = FAILURES_WIKI_DIR / "index.md"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("# Negative Knowledge Base (Failure Index)\n\n")
        f.write("This section tracks all factor candidates that failed the quality gate or execution process.\n\n")
        f.write("| Date | RunID | Factor Name | Type | Reason |\n")
        f.write("|------|-------|-------------|------|--------|\n")
        
        for i, fail in enumerate(failures[:200]): # 索引只展示最近 200 条
            name = fail.get('name', 'Unknown')
            run_id = fail.get('run_id', 'unknown')
            timestamp = fail.get('timestamp', 'unknown')[:10]
            fail_type = fail.get('type', 'unknown')
            reason = fail.get('reason', 'N/A').split('\n')[0][:50] + "..."
            
            # 对应的文件名链接
            safe_name = f"{timestamp.replace('-', '')}_{run_id}_{i}_{name.replace('/', '_')}.md"
            f.write(f"| {timestamp} | {run_id} | [{name}]({safe_name}) | {fail_type} | {reason} |\n")

if __name__ == "__main__":
    generate_failure_wiki()
