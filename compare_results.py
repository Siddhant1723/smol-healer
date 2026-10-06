"""
compare_results.py - Side-by-side evaluator for smol-healer.

Parses logs/baseline_results.json and logs/healed_results.json to report
per-task status shifts, latency differentials, and overall reliability gains.
"""

import json
import os
import sys

BASELINE_LOG = "logs/baseline_results.json"
HEALED_LOG = "logs/healed_results.json"


def load_log(path: str):
    if not os.path.exists(path):
        print(f"Error: Missing log file '{path}'. Ensure both runners have executed.")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    b_data = {row["task_id"]: row for row in load_log(BASELINE_LOG)}
    h_data = {row["task_id"]: row for row in load_log(HEALED_LOG)}

    tasks = list(b_data.keys())
    col_w = [18, 16, 22, 14, 12]
    header = (
        f"{'Task ID':<{col_w[0]}} | "
        f"{'Category':<{col_w[1]}} | "
        f"{'Baseline':<{col_w[2]}} | "
        f"{'Healed':<{col_w[3]}} | "
        f"{'Time Delta':<{col_w[4]}}"
    )
    divider = "-" * len(header)

    print("\n" + "=" * len(header))
    print("smol-healer BENCHMARK EVALUATION REPORT")
    print("=" * len(header))
    print(header)
    print(divider)

    b_passed = 0
    h_passed = 0
    b_total_time = 0.0
    h_total_time = 0.0
    recovered_tasks = []

    for tid in tasks:
        b = b_data.get(tid, {})
        h = h_data.get(tid, {})

        b_stat = b.get("status", "UNKNOWN")
        h_stat = h.get("status", "UNKNOWN")
        cat = b.get("category", "n/a")

        b_time = b.get("duration_seconds", 0.0)
        h_time = h.get("duration_seconds", 0.0)
        b_total_time += b_time
        h_total_time += h_time

        if b_stat == "PASSED":
            b_passed += 1
        if h_stat == "PASSED":
            h_passed += 1

        if b_stat != "PASSED" and h_stat == "PASSED":
            recovered_tasks.append(tid)

        delta = round(h_time - b_time, 2)
        delta_str = f"{delta:+}s" if delta != 0 else "0.0s"

        # Format baseline status display
        b_disp = "PASSED" if b_stat == "PASSED" else b_stat.replace("FAILED", "FAIL")

        print(
            f"{tid:<{col_w[0]}} | "
            f"{cat:<{col_w[1]}} | "
            f"{b_disp:<{col_w[2]}} | "
            f"{h_stat:<{col_w[3]}} | "
            f"{delta_str:<{col_w[4]}}"
        )

    print(divider)

    total_tasks = len(tasks)
    b_pct = round((b_passed / total_tasks) * 100, 1)
    h_pct = round((h_passed / total_tasks) * 100, 1)
    time_saved = round(b_total_time - h_total_time, 2)

    print(f"Accuracy Rate:     Baseline: {b_passed}/{total_tasks} ({b_pct}%)  -->  Healed: {h_passed}/{total_tasks} ({h_pct}%)")
    print(f"Total Execution:   Baseline: {round(b_total_time, 2)}s         -->  Healed: {round(h_total_time, 2)}s ({time_saved:+}s faster)")
    print(f"Recovered Tasks:   {len(recovered_tasks)} ({', '.join(recovered_tasks) if recovered_tasks else 'None'})")
    print("=" * len(header) + "\n")


if __name__ == "__main__":
    main()