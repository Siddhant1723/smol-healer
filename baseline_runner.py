"""
baseline_runner.py - Unpatched baseline evaluator for smol-healer.

Runs benchmark tasks against the stock Hugging Face smolagents CodeAgent
powered by local Ollama (qwen2.5:1.5b) without any runtime error-catching,
AST modifications, or type coercions. Records baseline failure metrics.
"""

import json
import os
import time
from smolagents import CodeAgent, LiteLLMModel
from tasks import calculate_vat, check_inventory, get_user_tier, BENCHMARK_TASKS

# 1. Configure the local LiteLLM provider for Ollama
MODEL_ID = "ollama_chat/qwen2.5:1.5b"
API_BASE = "http://localhost:11434"


def get_model() -> LiteLLMModel:
    """Returns an isolated LiteLLMModel connection."""
    return LiteLLMModel(
        model_id=MODEL_ID,
        api_base=API_BASE,
        num_ctx=4096,
        temperature=0.0,  # Greedy sampling for deterministic benchmarking
    )


def evaluate_output(actual, expected) -> bool:
    """Verifies whether actual agent output matches expected task output."""
    if actual is None:
        return False
    try:
        if isinstance(expected, (int, float)):
            return abs(float(actual) - float(expected)) < 1e-4
    except (ValueError, TypeError):
        pass
    return str(actual).strip() == str(expected).strip()


def run_baseline():
    os.makedirs("logs", exist_ok=True)
    results = []

    print("\n" + "=" * 65)
    print("SCRIPT 1: RUNNING RAW BASELINE (STOCK CODEAGENT)")
    print(f"Target: {MODEL_ID} via {API_BASE}")
    print("=" * 65 + "\n")

    for task in BENCHMARK_TASKS:
        task_id = task["id"]
        category = task["category"]
        prompt = task["prompt"]
        expected = task["expected"]

        print(f"[{task_id}] ({category})")
        print(f"  Prompt: {prompt}")

        # Create a fresh CodeAgent per task to prevent memory leakage
        agent = CodeAgent(
            tools=[calculate_vat, check_inventory, get_user_tier],
            model=get_model(),
            max_steps=4,
            verbosity_level=1,
        )

        start_time = time.time()
        actual_output = None
        error_name = None
        error_details = None
        task_status = "FAILED"

        try:
            raw_result = agent.run(prompt)
            actual_output = raw_result
            if evaluate_output(raw_result, expected):
                task_status = "PASSED"
            else:
                task_status = "FAILED (VALUE_MISMATCH)"
        except Exception as exc:
            error_name = type(exc).__name__
            error_details = str(exc)
            task_status = f"FAILED ({error_name})"

        elapsed = round(time.time() - start_time, 2)

        results.append({
            "task_id": task_id,
            "category": category,
            "target_vulnerability": task.get("target_vulnerability", ""),
            "prompt": prompt,
            "expected": expected,
            "actual_output": str(actual_output) if actual_output is not None else None,
            "status": task_status,
            "error_type": error_name,
            "error_message": error_details,
            "duration_seconds": elapsed,
        })

        print(f"  Result: {task_status} | Time: {elapsed}s")
        if actual_output is not None:
            print(f"  Output: {actual_output} (Expected: {expected})")
        if error_details:
            print(f"  Exception: {error_name}: {error_details[:100]}...")
        print("-" * 65)

    # Persist baseline logs for subsequent comparison
    log_path = "logs/baseline_results.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    passed_count = sum(1 for r in results if r["status"] == "PASSED")
    total_count = len(BENCHMARK_TASKS)

    print("\n" + "=" * 65)
    print(f"BASELINE SUMMARY: {passed_count}/{total_count} Passed ({round(passed_count / total_count * 100, 1)}%)")
    print(f"Empirical report saved to: {log_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    run_baseline()