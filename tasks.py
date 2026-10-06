"""
tasks.py - Benchmark task suite and mock tool definitions for smol-healer.

Defines deterministic tools with strict docstrings and type annotations
required by smolagents, paired with targeted evaluation tasks that expose
common failure modes in sub-3B parameter models.
"""

from smolagents import tool


# =====================================================================
# 1. Deterministic Mock Tools
# =====================================================================

@tool
def calculate_vat(price: float, rate: float = 0.20) -> float:
    """Calculates the Value Added Tax (VAT) for a given item price.

    Args:
        price: The pre-tax base price of the item.
        rate: VAT tax rate fraction (default is 0.20 for 20%).

    Returns:
        The computed VAT amount rounded to two decimal places.
    """
    return round(float(price) * float(rate), 2)


@tool
def check_inventory(sku: str) -> int:
    """Queries product stock inventory count for a specific SKU code.

    Args:
        sku: The product SKU string identifier (e.g., 'ITEM-A').

    Returns:
        The quantity of items in stock, or -1 if the SKU is not found.
    """
    inventory_db = {
        "ITEM-A": 15,
        "ITEM-B": 0,
        "ITEM-C": 42
    }
    return inventory_db.get(sku.strip().upper(), -1)


@tool
def get_user_tier(user_id: int) -> str:
    """Retrieves the customer loyalty tier associated with a user ID.

    Args:
        user_id: The numerical identifier for the user account. Must be an integer.

    Returns:
        The customer tier name ('Gold', 'Standard', 'VIP', or 'Unknown').
    """
    tiers = {
        101: "Gold",
        102: "Standard",
        103: "VIP"
    }
    # Strict lookup without defensive int() cast to test schema adherence
    return tiers.get(user_id, "Unknown")


# =====================================================================
# 2. Benchmark Task Suite
# =====================================================================

BENCHMARK_TASKS = [
    {
        "id": "T1_SINGLE_TOOL",
        "category": "tool_calling",
        "prompt": "Calculate the VAT on a 250 dollar item at 0.18 tax rate using calculate_vat.",
        "expected": 45.0,
        "eval_type": "exact_match",
        "target_vulnerability": "Baseline single tool invocation and argument passing"
    },
    {
        "id": "T2_CHAINED_TOOL",
        "category": "state_passing",
        "prompt": "Check the inventory count for SKU 'ITEM-A', multiply that inventory number by 4, and return the final total.",
        "expected": 60,
        "eval_type": "exact_match",
        "target_vulnerability": "Carrying state across tool outputs and local arithmetic"
    },
    {
        "id": "T3_CONDITIONAL_BRANCH",
        "category": "logic_branching",
        "prompt": "Check inventory for 'ITEM-B'. If it is 0, calculate the VAT on 150 with a 0.20 rate.",
        "expected": 30.0,
        "eval_type": "exact_match",
        "target_vulnerability": "Multi-step if/else control flow inside generated code"
    },
    {
        "id": "T4_UNAUTHORIZED_IMPORT",
        "category": "ast_security",
        "prompt": "Use the statistics module (import statistics) to calculate the mean of [10, 20, 30, 40].",
        "expected": 25.0,
        "eval_type": "exact_match",
        "target_vulnerability": "Triggers InterpreterError: statistics is standard library but blocked by default"
    },
    {
        "id": "T5_OMITTED_FINAL_ANSWER",
        "category": "execution_loop",
        "prompt": "Calculate 45 * 12 in Python and assign the result to a variable named answer.",
        "expected": 540,
        "eval_type": "exact_match",
        "target_vulnerability": "Missing final_answer() call: small models assign variables and loop until max_steps"
    },
    {
        "id": "T6_TYPE_COERCION",
        "category": "schema_validation",
        "prompt": "Get the membership tier for user ID '101' using get_user_tier.",
        "expected": "Gold",
        "eval_type": "exact_match",
        "target_vulnerability": "Type mismatch: passing string '101' returns 'Unknown' without schema coercion"
    }
]