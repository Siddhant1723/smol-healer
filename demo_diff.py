"""
demo_diff.py - Live terminal demonstration of baseline vs. healed harness.
"""

import json
import os

BASELINE_PATH = "logs/baseline_results.json"
HEALED_PATH = "logs/healed_results.json"


def show_diff():
    if not (os.path.exists(BASELINE_PATH) and os.path.exists(HEALED_PATH)):
        print("Missing log files. Run baseline_runner.py and resilient_runner.py first.")
        return

    with open(BASELINE_PATH) as f:
        b_data = {t["task_id"]: t for t in json.load(f)}
    with open(HEALED_PATH) as f:
        h_data = {t["task_id"]: t for t in json.load(f)}

    print("\n" + "=" * 80)
    print("smol-healer vs. STOCK SMOLAGENTS: ARCHITECTURAL DIFFERENCE REPORT")
    print("=" * 80)

    for tid, b in b_data.items():
        h = h_data.get(tid, {})
        print(f"\nTask: {tid} ({b.get('category')})")
        print(f"Prompt: \"{b.get('prompt')}\"")
        print("-" * 80)

        # Baseline details
        b_res = b.get("status")
        b_time = b.get("duration_seconds")
        b_out = b.get("actual_output")
        print(f"  [STOCK SMOLAGENTS]")
        print(f"    Status:   {b_res}")
        print(f"    Time:     {b_time}s")
        print(f"    Output:   {b_out[:70] if b_out else 'None'}...")

        # Healed details
        h_res = h.get("status")
        h_time = h.get("duration_seconds")
        h_out = h.get("actual_output")
        print(f"  [smol-healer]")
        print(f"    Status:   {h_res}")
        print(f"    Time:     {h_time}s (Delta: {round(h_time - b_time, 2):+}s)")
        print(f"    Output:   {h_out}")

        # Core difference highlight
        if b_res != h_res or abs(b_time - h_time) > 2.0:
            saved = round(b_time - h_time, 2)
            print(f"  >>> DIFFERENCE: Saved {saved}s of loop stall via AST/Tag Interception.")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    show_diff()