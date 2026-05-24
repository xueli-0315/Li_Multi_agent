from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from workflows import AlphaFactorMiningWorkflow


def main() -> None:
    workflow = AlphaFactorMiningWorkflow()
    trace = workflow.run({"market": "crypto_cross_section", "scenario": "daily_factor_research"})
    output = {
        "record_count": len(trace.records),
        "agents": [record.agent_name for record in trace.records],
        "final_shared_context": trace.final_shared_context,
        "errors": [record.error for record in trace.records if record.error is not None],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
