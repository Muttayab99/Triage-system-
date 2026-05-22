from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils import write_json


def apply_routing(predictions: List[Dict[str, Any]], routing_rules: Dict[str, str]) -> None:
    for pred in predictions:
        category = pred.get("category", "other")
        pred["route_to"] = routing_rules.get(category, routing_rules.get("other", "manual_review_queue"))


def build_final_queue(
    predictions: List[Dict[str, Any]],
    overrides: List[Dict[str, Any]],
    routing_rules: Dict[str, str],
    out_path: Path,
) -> List[Dict[str, Any]]:
    apply_routing(predictions, routing_rules)
    overridden_ids = {o["ticket_id"] for o in overrides}
    final_queue = []
    for pred in predictions:
        ticket_id = pred["ticket_id"]
        final_queue.append(
            {
                "ticket_id": ticket_id,
                "final_category": pred["category"],
                "final_priority": pred["priority"],
                "final_route_to": pred["route_to"],
                "suggested_reply": pred["suggested_reply"],
                "was_overridden": ticket_id in overridden_ids,
            }
        )
    write_json(out_path, final_queue)
    return final_queue


def build_queue_summary(
    final_queue: List[Dict[str, Any]],
    overrides: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    total = len(final_queue)
    by_category = Counter([q["final_category"] for q in final_queue])
    by_priority = Counter([q["final_priority"] for q in final_queue])
    by_route = Counter([q["final_route_to"] for q in final_queue])

    lines = [
        "# Queue Summary",
        "",
        f"Total tickets: {total}",
        "",
        "## Count by category",
    ]
    for key, value in sorted(by_category.items()):
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Count by priority"])
    for key, value in sorted(by_priority.items()):
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Queue breakdown"])
    for key, value in sorted(by_route.items()):
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Overrides"])
    if not overrides:
        lines.append("- None")
    else:
        for o in overrides:
            lines.append(
                f"- {o['ticket_id']}: {o['old_category']}->{o['new_category']}, {o['old_priority']}->{o['new_priority']}"
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")
