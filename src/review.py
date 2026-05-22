from typing import Any, Dict, List, Tuple


def collect_overrides(
    predictions: List[Dict[str, Any]],
    allowed_categories: List[str],
    allowed_priorities: List[str],
) -> List[Dict[str, Any]]:
    by_id = {p["ticket_id"]: p for p in predictions}
    overrides: List[Dict[str, Any]] = []

    print("\nPredicted categories and priorities:")
    for p in predictions:
        print(f"- {p['ticket_id']}: {p['category']} / {p['priority']}")

    print("\nEnter any overrides as: ticket_id,category,priority")
    print("Press Enter on an empty line when done.")

    while True:
        line = input("> ").strip()
        if not line:
            break
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            print("Invalid format. Use: ticket_id,category,priority")
            continue
        ticket_id, new_category, new_priority = parts
        if ticket_id not in by_id:
            print("Unknown ticket_id")
            continue
        if new_category not in allowed_categories:
            print("Invalid category")
            continue
        if new_priority not in allowed_priorities:
            print("Invalid priority")
            continue
        old_category = by_id[ticket_id]["category"]
        old_priority = by_id[ticket_id]["priority"]
        if old_category == new_category and old_priority == new_priority:
            print("No change from existing values.")
            continue
        overrides.append(
            {
                "ticket_id": ticket_id,
                "old_category": old_category,
                "new_category": new_category,
                "old_priority": old_priority,
                "new_priority": new_priority,
            }
        )
        by_id[ticket_id]["category"] = new_category
        by_id[ticket_id]["priority"] = new_priority

    return overrides
