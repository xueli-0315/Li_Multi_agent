import sys
import os
import json
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

# Load .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            if "=" in line and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value.strip("'\"")

# Infrastructure imports
from llm import LLMGateway, GatewayModelClient, RetryPolicy
from llm.providers import OpenAICompatibleProvider

KNOWLEDGE_DIR = PROJECT_ROOT / "factor_library" / "raw" / "negative_knowledge"
FAILURES_FILE = KNOWLEDGE_DIR / "all_failures.jsonl"
LESSONS_FILE = KNOWLEDGE_DIR / "distilled_lessons.md"

def distill_knowledge():
    if not FAILURES_FILE.exists():
        print("No failures recorded yet.")
        return

    failures = []
    with open(FAILURES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            failures.append(json.loads(line))

    if not failures:
        print("Failures file is empty.")
        return

    # 抽取最近的 50 条失败记录进行复盘 (确保包含下午最新的尝试)
    recent_failures = failures[-50:]
    
    # Prepare the context for LLM
    failure_context = ""
    for f in recent_failures:
        failure_context += f"- Type: {f.get('type', 'unknown')}, Factor: {f.get('name')}, Reason: {f.get('reason')}, Expr: {f.get('expression', 'N/A')}\n"

    # Initialize LLM Client
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("LLM_MODEL", "gpt-4-turbo")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment.")
        return

    provider = OpenAICompatibleProvider(api_key=api_key, base_url=base_url)
    gateway = LLMGateway(provider=provider, retry_policy=RetryPolicy(), default_model=model_name)
    client = GatewayModelClient(gateway)

    system_prompt = """You are a senior Quantitative Researcher. 
Your task is to analyze a list of failed alpha factor ideas and 'distill' them into 3-5 concise, high-level lessons.
Group the failures by logic patterns (e.g., 'Short-term price action often leads to high turnover').
Focus on specifically identifying issues with recently attempted logic like signed deltas or volume z-scores if they appear in the data.
Format your response as a clean Markdown document."""

    user_prompt = f"Here are the recent 50 failed attempts in our factor mining pipeline:\n\n{failure_context}\n\nPlease provide the distilled lessons in Markdown."

    print("Distilling knowledge using LLM...")
    try:
        distilled_text = client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        
        with open(LESSONS_FILE, "w", encoding="utf-8") as f:
            import datetime
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"# 因子挖掘负面知识库 (Distilled Lessons)\n\n")
            f.write(f"> 上次更新时间: {now_str}\n\n")
            f.write(distilled_text)
        
        print(f"Successfully updated {LESSONS_FILE}")
    except Exception as e:
        print(f"Distillation failed: {e}")

if __name__ == "__main__":
    distill_knowledge()
