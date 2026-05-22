import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from main import StageTracker, build_escalations
from src.normalize import normalize_tickets
from src.outputs import build_final_queue, build_queue_summary
from src.triage import predict_triage
from src.utils import get_env, now_iso, read_json, sha256_json, write_json
from validate import validate_all


st.set_page_config(page_title="Support Triage Pipeline", layout="wide")


st.title("Support Triage Pipeline")
st.caption("Run normalization, LLM triage, human review, and final queue generation.")


with st.sidebar:
    st.header("Inputs")
    tickets_path = st.text_input("tickets.json path", value="input/tickets.json")
    config_path = st.text_input("triage_config.json path", value="input/triage_config.json")
    out_dir = st.text_input("Output directory", value="output")
    st.header("Auth")
    api_key = st.text_input("GROQ API Key (optional)", type="password")


base_dir = Path(__file__).parent
env_file = base_dir / ".env"
resolved_tickets = (base_dir / tickets_path).resolve()
resolved_config = (base_dir / config_path).resolve()
resolved_out_dir = (base_dir / out_dir).resolve()
resolved_out_dir.mkdir(parents=True, exist_ok=True)


if "predictions" not in st.session_state:
    st.session_state["predictions"] = []
if "normalized" not in st.session_state:
    st.session_state["normalized"] = []
if "config" not in st.session_state:
    st.session_state["config"] = {}
if "overrides" not in st.session_state:
    st.session_state["overrides"] = []


col1, col2 = st.columns(2)

with col1:
    if st.button("Run normalization + triage", type="primary"):
        if not resolved_tickets.exists() or not resolved_config.exists():
            st.error("Input files not found. Check paths.")
        else:
            if api_key:
                os.environ["GROQ_API_KEY"] = api_key
            resolved_key = get_env("GROQ_API_KEY", ["groq_api"], env_file=env_file)
            if resolved_key:
                os.environ["GROQ_API_KEY"] = resolved_key
            tickets = read_json(resolved_tickets)
            config = read_json(resolved_config)
            model = get_env("model", env_file=env_file) or "openai/gpt-oss-120b"

            tracker = StageTracker(resolved_out_dir)
            tracker.advance("INIT")

            write_json(
                resolved_out_dir / "run_manifest.json",
                {
                    "inputs": [resolved_tickets.name, resolved_config.name],
                    "timestamp": now_iso(),
                    "ticket_hash": sha256_json(tickets),
                    "config_hash": sha256_json(config),
                },
            )
            tracker.advance("INPUTS_LOADED")

            normalized = normalize_tickets(tickets, resolved_out_dir / "normalized_tickets.json")
            tracker.advance("TICKETS_NORMALIZED")

            try:
                predictions = predict_triage(
                    normalized,
                    config,
                    out_path=resolved_out_dir / "triage_predictions.json",
                    llm_log_path=resolved_out_dir / "llm_calls.jsonl",
                    error_log_path=resolved_out_dir / "triage_errors.jsonl",
                    model=model,
                    api_key=os.environ.get("GROQ_API_KEY", ""),
                    batch_size=0,
                    retries=2,
                )
                tracker.advance("TRIAGE_PREDICTED")

                st.session_state["normalized"] = normalized
                st.session_state["predictions"] = predictions
                st.session_state["config"] = config
                st.session_state["overrides"] = []

                st.success("Triage predictions generated.")
            except Exception as exc:
                st.error(f"Triage failed: {exc}")

with col2:
    if st.button("Validate outputs"):
        errors = validate_all(resolved_out_dir, input_dir=base_dir / "input")
        if errors:
            st.error("Validation failed")
            st.write(errors)
        else:
            st.success("Validation passed")


st.subheader("Review and finalize")
if st.session_state["predictions"]:
    allowed_categories = st.session_state["config"].get("allowed_categories", [])
    allowed_priorities = st.session_state["config"].get("allowed_priorities", [])

    df = pd.DataFrame(
        [
            {
                "ticket_id": p["ticket_id"],
                "category": p["category"],
                "priority": p["priority"],
                "route_to": p.get("route_to", ""),
                "confidence": p.get("confidence", 0.0),
            }
            for p in st.session_state["predictions"]
        ]
    )

    edited_df = st.data_editor(
        df,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "category": st.column_config.SelectboxColumn("category", options=allowed_categories),
            "priority": st.column_config.SelectboxColumn("priority", options=allowed_priorities),
        },
    )

    if st.button("Apply overrides + finalize", type="primary"):
        overrides: List[Dict[str, Any]] = []
        by_id = {p["ticket_id"]: p for p in st.session_state["predictions"]}
        for row in edited_df.to_dict(orient="records"):
            ticket_id = row["ticket_id"]
            if ticket_id not in by_id:
                continue
            pred = by_id[ticket_id]
            if row["category"] != pred["category"] or row["priority"] != pred["priority"]:
                overrides.append(
                    {
                        "ticket_id": ticket_id,
                        "old_category": pred["category"],
                        "new_category": row["category"],
                        "old_priority": pred["priority"],
                        "new_priority": row["priority"],
                    }
                )
                pred["category"] = row["category"]
                pred["priority"] = row["priority"]

        write_json(resolved_out_dir / "review_overrides.json", overrides)

        final_queue = build_final_queue(
            st.session_state["predictions"],
            overrides,
            st.session_state["config"].get("routing_rules", {}),
            resolved_out_dir / "final_queue.json",
        )
        build_queue_summary(final_queue, overrides, resolved_out_dir / "queue_summary.md")

        escalations = build_escalations(st.session_state["predictions"], resolved_out_dir / "escalations.json")
        write_json(
            resolved_out_dir / "run_summary.json",
            {
                "ticket_count": len(st.session_state["predictions"]),
                "override_count": len(overrides),
                "escalation_count": len(escalations),
                "output_dir": str(resolved_out_dir),
                "model": model,
            },
        )

        tracker = StageTracker(resolved_out_dir)
        tracker.advance("INIT")
        tracker.advance("INPUTS_LOADED")
        tracker.advance("TICKETS_NORMALIZED")
        tracker.advance("TRIAGE_PREDICTED")
        tracker.advance("HUMAN_REVIEW_COMPLETE")
        tracker.advance("FINAL_QUEUE_GENERATED")
        tracker.advance("VALIDATION_COMPLETE")
        tracker.advance("RESULTS_FINALISED")

        st.session_state["overrides"] = overrides
        st.success("Final outputs generated.")
else:
    st.info("Run normalization + triage to see predictions.")

st.subheader("Outputs")
summary_path = resolved_out_dir / "queue_summary.md"
final_queue_path = resolved_out_dir / "final_queue.json"
predictions_path = resolved_out_dir / "triage_predictions.json"

metrics = st.columns(4)
metrics[0].metric("Tickets", len(st.session_state["predictions"]))
metrics[1].metric("Overrides", len(st.session_state["overrides"]))
metrics[2].metric("Escalations", len(read_json(resolved_out_dir / "escalations.json")) if (resolved_out_dir / "escalations.json").exists() else 0)
metrics[3].metric("Artifacts", len([p for p in resolved_out_dir.iterdir() if p.is_file()]))

tabs = st.tabs(["Queue Summary", "Final Queue", "Predictions", "Artifacts"])
with tabs[0]:
    if summary_path.exists():
        st.markdown(summary_path.read_text(encoding="utf-8"))
    else:
        st.info("Queue summary will appear after finalization.")
with tabs[1]:
    if final_queue_path.exists():
        st.dataframe(read_json(final_queue_path), use_container_width=True)
    else:
        st.info("Final queue will appear after finalization.")
with tabs[2]:
    if predictions_path.exists():
        st.dataframe(read_json(predictions_path), use_container_width=True)
    else:
        st.info("Predictions will appear after triage.")
with tabs[3]:
    st.write("Artifacts generated in output directory:")
    st.write([p.name for p in resolved_out_dir.iterdir() if p.is_file()])

st.subheader("Artifacts")
artifacts = [
    "normalized_tickets.json",
    "triage_predictions.json",
    "review_overrides.json",
    "final_queue.json",
    "queue_summary.md",
    "escalations.json",
    "llm_calls.jsonl",
    "triage_errors.jsonl",
    "run_manifest.json",
    "run_summary.json",
    "pipeline_state.json",
]

existing = [name for name in artifacts if (resolved_out_dir / name).exists()]
if existing:
    st.write("Generated artifacts:")
    st.write(existing)
