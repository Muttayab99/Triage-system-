import argparse
import platform
from pathlib import Path
from typing import Any, Dict, List

from src.normalize import normalize_tickets
from src.outputs import build_final_queue, build_queue_summary
from src.review import collect_overrides
from src.triage import predict_triage
from src.utils import get_env, now_iso, read_json, sha256_json, write_json
from validate import validate_all


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


class StageTracker:
	def __init__(self, base_dir: Path) -> None:
		self.base_dir = base_dir
		self.stage_index = -1
		self.state_path = base_dir / "pipeline_state.json"
		self.state: Dict[str, Any] = {"stages": []}

	def current_stage(self) -> str:
		if self.stage_index < 0:
			return ""
		return STAGE_ORDER[self.stage_index]

	def require(self, stage: str) -> None:
		if self.current_stage() != stage:
			raise RuntimeError(f"Stage gate failed: expected {stage}, got {self.current_stage()}")

	def advance(self, stage: str) -> None:
		if stage not in STAGE_ORDER:
			raise ValueError(f"Unknown stage: {stage}")
		expected = STAGE_ORDER[self.stage_index + 1]
		if stage != expected:
			raise ValueError(f"Invalid stage transition: expected {expected}, got {stage}")
		self.stage_index += 1
		self.state["stages"].append({"stage": stage, "timestamp": now_iso()})
		write_json(self.state_path, self.state)


def build_escalations(predictions: List[Dict[str, Any]], out_path: Path) -> List[Dict[str, Any]]:
	escalations = []
	for pred in predictions:
		confidence = float(pred.get("confidence", 0.0))
		if pred.get("category") == "other" or confidence < 0.60:
			escalations.append(
				{
					"ticket_id": pred.get("ticket_id"),
					"category": pred.get("category"),
					"priority": pred.get("priority"),
					"confidence": confidence,
				}
			)
	write_json(out_path, escalations)
	return escalations


def main() -> int:
	parser = argparse.ArgumentParser(description="Support triage pipeline")
	parser.add_argument("--tickets", default="input/tickets.json", help="Path to tickets.json")
	parser.add_argument("--config", default="input/triage_config.json", help="Path to triage_config.json")
	parser.add_argument("--model", default=None, help="LLM model name")
	parser.add_argument("--out-dir", default="output", help="Output directory (default: output)")
	parser.add_argument("--env-file", default=".env", help="Env file path (set to empty to disable)")
	parser.add_argument("--batch-size", type=int, default=0, help="Tickets per LLM call (0 for all)")
	parser.add_argument("--retries", type=int, default=2, help="LLM retry attempts")
	parser.add_argument("--skip-validate", action="store_true", help="Skip validation step")
	args = parser.parse_args()

	base_dir = Path(__file__).parent
	out_dir = (base_dir / args.out_dir).resolve()
	out_dir.mkdir(parents=True, exist_ok=True)
	tracker = StageTracker(out_dir)
	tracker.advance("INIT")

	tickets_path = base_dir / args.tickets
	config_path = base_dir / args.config
	normalized_path = out_dir / "normalized_tickets.json"
	predictions_path = out_dir / "triage_predictions.json"
	overrides_path = out_dir / "review_overrides.json"
	final_queue_path = out_dir / "final_queue.json"
	summary_path = out_dir / "queue_summary.md"
	llm_log_path = out_dir / "llm_calls.jsonl"
	error_log_path = out_dir / "triage_errors.jsonl"
	escalations_path = out_dir / "escalations.json"
	run_manifest_path = out_dir / "run_manifest.json"
	run_summary_path = out_dir / "run_summary.json"

	tickets = read_json(tickets_path)
	config = read_json(config_path)
	normalized_hash = sha256_json(tickets)
	config_hash = sha256_json(config)
	manifest = {
		"inputs": [str(tickets_path.name), str(config_path.name)],
		"timestamp": now_iso(),
		"ticket_hash": normalized_hash,
		"config_hash": config_hash,
		"python_version": platform.python_version(),
		"platform": platform.platform(),
	}
	write_json(run_manifest_path, manifest)
	tracker.advance("INPUTS_LOADED")

	normalized = normalize_tickets(tickets, normalized_path)
	manifest["normalized_at"] = now_iso()
	write_json(run_manifest_path, manifest)
	tracker.advance("TICKETS_NORMALIZED")

	env_file = None
	if args.env_file:
		env_file = Path(args.env_file)
	api_key = get_env("GROQ_API_KEY", ["groq_api"], env_file=env_file)
	if not api_key:
		raise RuntimeError("Missing GROQ_API_KEY (or groq_api) in environment")
	model = args.model or get_env("model", env_file=env_file) or "openai/gpt-oss-120b"

	predictions = predict_triage(
		normalized,
		config,
		out_path=predictions_path,
		llm_log_path=llm_log_path,
		error_log_path=error_log_path,
		model=model,
		api_key=api_key,
		batch_size=args.batch_size,
		retries=args.retries,
	)
	tracker.advance("TRIAGE_PREDICTED")

	overrides = collect_overrides(
		predictions,
		config.get("allowed_categories", []),
		config.get("allowed_priorities", []),
	)
	write_json(overrides_path, overrides)
	tracker.advance("HUMAN_REVIEW_COMPLETE")

	tracker.require("HUMAN_REVIEW_COMPLETE")
	final_queue = build_final_queue(
		predictions,
		overrides,
		config.get("routing_rules", {}),
		final_queue_path,
	)
	build_queue_summary(final_queue, overrides, summary_path)
	tracker.advance("FINAL_QUEUE_GENERATED")

	escalations = build_escalations(predictions, escalations_path)
	write_json(
		run_summary_path,
		{
			"ticket_count": len(tickets),
			"override_count": len(overrides),
			"escalation_count": len(escalations),
			"output_dir": str(out_dir),
			"model": model,
		},
	)

	if not args.skip_validate:
		errors = validate_all(out_dir, input_dir=base_dir / "input")
		if errors:
			raise RuntimeError("Validation failed: " + "; ".join(errors))
	tracker.advance("VALIDATION_COMPLETE")
	tracker.advance("RESULTS_FINALISED")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
