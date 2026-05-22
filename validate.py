import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from src.utils import read_json, word_count

REQUIRED_ARTIFACTS = [
    "normalized_tickets.json",
    "triage_predictions.json",
    "review_overrides.json",
    "final_queue.json",
    "queue_summary.md",
    "run_manifest.json",
    "run_summary.json",
]

STAGE_ORDER = [
    "INIT",
    "INPUTS_LOADED",
    "TICKETS_NORMALIZED",
    "TRIAGE_PREDICTED",
    "HUMAN_REVIEW_COMPLETE",
    "FINAL_QUEUE_GENERATED",
    "VALIDATION_COMPLETE",
    "RESULTS_FINALISED",
]


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _error(errors: List[str], message: str) -> None:
    errors.append(message)


def validate_artifacts(base_dir: Path) -> List[str]:
    errors: List[str] = []

    for name in REQUIRED_ARTIFACTS:
        if not (base_dir / name).exists():
            _error(errors, f"Missing artifact: {name}")

    return errors


def validate_pipeline_state(base_dir: Path) -> List[str]:
    errors: List[str] = []
    state_path = base_dir / "pipeline_state.json"
    if not state_path.exists():
        return errors

    state = _read_json(state_path)
    observed = [entry.get("stage") for entry in state.get("stages", [])]
    expected_index = 0
    for stage in observed:
        if expected_index < len(STAGE_ORDER) and stage == STAGE_ORDER[expected_index]:
            expected_index += 1
    if expected_index != len(STAGE_ORDER):
        _error(errors, "Pipeline stages are missing or out of order")
    return errors


def validate_predictions(base_dir: Path, input_dir: Path) -> List[str]:
    errors: List[str] = []
    config = read_json(input_dir / "triage_config.json")
    allowed_categories = set(config.get("allowed_categories", []))
    allowed_priorities = set(config.get("allowed_priorities", []))
    routing_rules = config.get("routing_rules", {})
    reply_style = config.get("reply_style", {})
    max_words = int(reply_style.get("max_words", 80))

    tickets = read_json(input_dir / "tickets.json")
    predictions = read_json(base_dir / "triage_predictions.json")

    if len(predictions) != len(tickets):
        _error(errors, "Predictions count does not match ticket count")

    ticket_ids = {t.get("ticket_id") for t in tickets}
    pred_ids = {p.get("ticket_id") for p in predictions}
    if ticket_ids != pred_ids:
        _error(errors, "Prediction ticket IDs do not match input tickets")

    for pred in predictions:
        category = pred.get("category")
        priority = pred.get("priority")
        route_to = pred.get("route_to")
        reply = pred.get("suggested_reply", "")

        if category not in allowed_categories:
            _error(errors, f"Invalid category: {category}")
        if priority not in allowed_priorities:
            _error(errors, f"Invalid priority: {priority}")
        if routing_rules.get(category) != route_to:
            _error(errors, f"Routing mismatch for {pred.get('ticket_id')}")
        if word_count(reply) > max_words:
            _error(errors, f"Reply too long for {pred.get('ticket_id')}")

    return errors


def validate_overrides(base_dir: Path, input_dir: Path) -> List[str]:
    errors: List[str] = []
    overrides = read_json(base_dir / "review_overrides.json")
    config = read_json(input_dir / "triage_config.json")
    allowed_categories = set(config.get("allowed_categories", []))
    allowed_priorities = set(config.get("allowed_priorities", []))

    for o in overrides:
        if o.get("new_category") not in allowed_categories:
            _error(errors, "Override contains invalid category")
        if o.get("new_priority") not in allowed_priorities:
            _error(errors, "Override contains invalid priority")

    return errors


def validate_inputs_read(base_dir: Path) -> List[str]:
    errors: List[str] = []
    manifest_path = base_dir / "run_manifest.json"
    if not manifest_path.exists():
        _error(errors, "Missing run_manifest.json")
        return errors
    manifest = read_json(manifest_path)
    inputs = set(manifest.get("inputs", []))
    if "tickets.json" not in inputs:
        _error(errors, "run_manifest.json does not list tickets.json")
    if "triage_config.json" not in inputs:
        _error(errors, "run_manifest.json does not list triage_config.json")
    return errors


def validate_final_queue(base_dir: Path, input_dir: Path) -> List[str]:
    errors: List[str] = []
    config = read_json(input_dir / "triage_config.json")
    routing_rules = config.get("routing_rules", {})

    final_queue = read_json(base_dir / "final_queue.json")
    for item in final_queue:
        category = item.get("final_category")
        route_to = item.get("final_route_to")
        if routing_rules.get(category) != route_to:
            _error(errors, f"Final routing mismatch for {item.get('ticket_id')}")
    return errors


def validate_overrides_applied(base_dir: Path, input_dir: Path) -> List[str]:
    errors: List[str] = []
    overrides = read_json(base_dir / "review_overrides.json")
    if not overrides:
        return errors
    config = read_json(input_dir / "triage_config.json")
    routing_rules = config.get("routing_rules", {})
    final_queue = read_json(base_dir / "final_queue.json")
    by_id = {item.get("ticket_id"): item for item in final_queue}
    for override in overrides:
        ticket_id = override.get("ticket_id")
        item = by_id.get(ticket_id)
        if not item:
            _error(errors, f"Override ticket missing in final_queue.json: {ticket_id}")
            continue
        if item.get("final_category") != override.get("new_category"):
            _error(errors, f"Override category not applied for {ticket_id}")
        if item.get("final_priority") != override.get("new_priority"):
            _error(errors, f"Override priority not applied for {ticket_id}")
        expected_route = routing_rules.get(override.get("new_category"))
        if expected_route and item.get("final_route_to") != expected_route:
            _error(errors, f"Override route not applied for {ticket_id}")
        if item.get("was_overridden") is not True:
            _error(errors, f"was_overridden not true for {ticket_id}")
    return errors


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def validate_normalization_before_llm(base_dir: Path) -> List[str]:
    errors: List[str] = []
    manifest_path = base_dir / "run_manifest.json"
    llm_log_path = base_dir / "llm_calls.jsonl"
    if not manifest_path.exists() or not llm_log_path.exists():
        return errors

    manifest = read_json(manifest_path)
    normalized_at = manifest.get("normalized_at")
    if not normalized_at:
        _error(errors, "run_manifest.json missing normalized_at")
        return errors

    logs = _read_jsonl(llm_log_path)
    if not logs:
        _error(errors, "llm_calls.jsonl is empty")
        return errors

    first_call = logs[0].get("timestamp")
    if not first_call:
        _error(errors, "llm_calls.jsonl missing timestamp")
        return errors

    try:
        normalized_dt = datetime.fromisoformat(normalized_at)
        llm_dt = datetime.fromisoformat(first_call)
    except Exception:
        _error(errors, "Invalid ISO timestamps for normalization/LLM call")
        return errors

    if normalized_dt > llm_dt:
        _error(errors, "Normalization happened after LLM call")
    return errors


def validate_all(base_dir: Path, input_dir: Path) -> List[str]:
    errors: List[str] = []
    errors.extend(validate_artifacts(base_dir))
    errors.extend(validate_pipeline_state(base_dir))
    errors.extend(validate_inputs_read(base_dir))
    errors.extend(validate_predictions(base_dir, input_dir))
    errors.extend(validate_overrides(base_dir, input_dir))
    errors.extend(validate_final_queue(base_dir, input_dir))
    errors.extend(validate_overrides_applied(base_dir, input_dir))
    errors.extend(validate_normalization_before_llm(base_dir))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate triage pipeline outputs")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: project root)")
    parser.add_argument("--input-dir", default="input", help="Input directory (default: input)")
    args = parser.parse_args()
    base_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent / "output"
    input_dir = Path(args.input_dir) if args.input_dir else Path(__file__).parent / "input"
    errors = validate_all(base_dir, input_dir=input_dir)
    if errors:
        print("Validation failed:")
        for err in errors:
            print(f"- {err}")
        return 1
    print("Validation passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
