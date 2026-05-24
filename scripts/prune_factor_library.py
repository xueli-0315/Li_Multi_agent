import json
import os
import shutil
from datetime import datetime
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = PROJECT_ROOT / "logs" / "alpha_factor_mining_loop"
LIB_DIR = PROJECT_ROOT / "factor_library" / "raw"
RAW_JSON = LIB_DIR / "all_factors_library.json"
CODE_DIR = LIB_DIR / "factor_codes"

def prune():
    print("Starting deep synchronization from logs...")
    
    # 1. Ingest ALL factors from logs
    synced_records = []
    seen_keys = set()
    
    if not LOGS_DIR.exists():
        print(f"Error: Logs directory {LOGS_DIR} not found.")
        return

    # Sort run dirs by time to keep things ordered
    run_dirs = sorted([d for d in LOGS_DIR.iterdir() if d.is_dir()], key=lambda x: x.name)
    
    for run_dir in run_dirs:
        run_id = run_dir.name
        # Sort loop files
        loop_files = sorted(run_dir.glob("loop_*.json"), key=lambda x: x.name)
        
        for loop_file in loop_files:
            try:
                with open(loop_file, 'r', encoding='utf-8') as f:
                    loop_data = json.load(f)
                
                # 1. Root metadata
                loop = loop_data.get("loop_index", 0)
                
                # 2. Pre-scan to collect context (hypothesis & specs)
                current_hypothesis = "No hypothesis found"
                current_specs = {}
                
                for record in loop_data.get("records", []):
                    updates = record.get("output_snapshot", {}).get("shared_updates", {})
                    # Hypothesis check
                    if "hypothesis" in updates:
                        current_hypothesis = updates["hypothesis"]
                    # Experiment specs (contains expressions)
                    exp_spec = updates.get("experiment_spec", {})
                    if exp_spec:
                        for f in exp_spec.get("factors", []):
                            current_specs[f["factor_name"]] = f

                # 3. Process metrics records
                name_to_code = {}
                
                # Deep recursive search for codes in the entire loop_data
                def find_codes(obj):
                    if isinstance(obj, dict):
                        if "factor_name" in obj and "code" in obj:
                            name_to_code[obj["factor_name"]] = obj["code"]
                        elif "name" in obj and "code" in obj:
                            name_to_code[obj["name"]] = obj["code"]
                        for v in obj.values():
                            find_codes(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            find_codes(item)
                
                find_codes(loop_data)

                for record in loop_data.get("records", []):
                    updates = record.get("output_snapshot", {}).get("shared_updates", {})
                    bt_report = updates.get("backtest_report", {})
                    per_metrics = bt_report.get("per_factor_metrics", {})
                    
                    if not per_metrics:
                        continue
                        
                    # Extract rejected names
                    failed_details = bt_report.get("failed_details", [])
                    if not failed_details:
                        failed_details = updates.get("failed_details", [])
                    rejected_names = {d["name"] for d in failed_details if "name" in d}

                    for f_idx, (f_name, metrics) in enumerate(per_metrics.items()):
                        # Filter...
                        if f_name in rejected_names: continue
                        ic_val = abs(metrics.get("IC", 0))
                        if ic_val < 0.003: continue
                        
                        key = (run_id, int(loop), f_name)
                        if key in seen_keys: continue
                        
                        spec = current_specs.get(f_name, {})
                        
                        # [RECOVERY LOGIC] 
                        # Restore code from our deep-scanned cache with standard prefix
                        code_content = name_to_code.get(f_name)
                        standard_name = f"{run_id}_L{loop}_I{f_idx}_{f_name}"
                        if code_content:
                            CODE_DIR.mkdir(parents=True, exist_ok=True)
                            target_py = CODE_DIR / f"{standard_name}.py"
                            if not target_py.exists():
                                with open(target_py, "w", encoding="utf-8") as f_code:
                                    f_code.write(code_content)
                                print(f"Restored code for: {standard_name}")

                        record_to_add = {
                            "run_id": run_id,
                            "loop_round": int(loop),
                            "intra_loop_index": f_idx,
                            "factor_name": f_name,
                            "factor_file": f"{standard_name}.py" if code_content else f"{f_name}.py",
                            "factor_expression": spec.get("expression", "Unknown"),
                            "hypothesis": current_hypothesis,
                            "metrics": metrics
                        }
                        synced_records.append(record_to_add)
                        seen_keys.add(key)
                                
            except Exception as e:
                print(f"Skipping {loop_file} due to error: {e}")

    print(f"Extracted {len(synced_records)} unique factors from logs.")

    if not synced_records:
        print("No valid factors found in logs. Aborting to prevent data loss.")
        return

    # 2. Rebuild JSON library
    lib_data = {
        "metadata": {
            "last_updated": datetime.now().isoformat(),
            "total_records": len(synced_records),
            "version": "2.1-synced"
        },
        "records": synced_records
    }
    
    # Backup existing
    if RAW_JSON.exists():
        backup_path = RAW_JSON.with_suffix(".json.bak")
        shutil.copy(RAW_JSON, backup_path)
        print(f"Existing library backed up to {backup_path}")

    with open(RAW_JSON, 'w', encoding='utf-8') as f:
        json.dump(lib_data, f, indent=2, ensure_ascii=False)
    print(f"Successfully rebuilt {RAW_JSON} with {len(synced_records)} factors.")

    # 3. Clean up factor_codes directory
    if CODE_DIR.exists():
        valid_filenames = {r.get("factor_file") for r in synced_records if r.get("factor_file")}
        deleted_codes = 0
        for code_file in CODE_DIR.glob("*.py"):
            if code_file.name not in valid_filenames:
                code_file.unlink()
                deleted_codes += 1
        print(f"Deleted {deleted_codes} orphaned code files.")

    # 4. Re-build Wiki
    print("Re-building Wiki from synchronized library...")
    try:
        import subprocess
        wiki_script = PROJECT_ROOT / "scripts" / "build_factor_wiki.py"
        subprocess.run(["python3", str(wiki_script)], check=True)
        print("Wiki rebuilt successfully.")
    except Exception as e:
        print(f"Wiki rebuild failed: {e}")

if __name__ == "__main__":
    prune()
