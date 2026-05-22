import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils import now_iso, sha256_text, truncate_words, word_count, write_json, write_jsonl


class TriageError(Exception):
    pass


def build_prompt(config: Dict[str, Any], normalized: List[Dict[str, Any]]) -> str:
    allowed_categories = config.get("allowed_categories", [])
    allowed_priorities = config.get("allowed_priorities", [])
    routing_rules = config.get("routing_rules", {})
    reply_style = config.get("reply_style", {})

    instructions = {
        "task": "Classify and draft replies for support tickets.",
        "allowed_categories": allowed_categories,
        "allowed_priorities": allowed_priorities,
        "routing_rules": routing_rules,
        "reply_style": reply_style,
        "output_schema": {
            "ticket_id": "string",
            "category": "string",
            "priority": "string",
            "reason": "string",
            "suggested_reply": "string",
            "route_to": "string",
            "confidence": "number (0.0-1.0)",
        },
        "output_format": "Return a JSON array only, with one object per ticket.",
    }

    payload = {
        "instructions": instructions,
        "tickets": normalized,
    }

    return json.dumps(payload, ensure_ascii=True, indent=2)


def _extract_json_array(text: str) -> List[Dict[str, Any]]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        stripped = stripped.replace("json", "").strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        return json.loads(stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        parsed = json.loads(stripped)
        if isinstance(parsed, dict) and "predictions" in parsed:
            return parsed["predictions"]
    left = stripped.find("[")
    right = stripped.rfind("]")
    if left != -1 and right != -1 and left < right:
        return json.loads(stripped[left : right + 1])
    return []


def call_groq(prompt: str, model: str, api_key: str) -> Tuple[str, Dict[str, Any]]:
    try:
        from groq import Groq
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise TriageError(f"groq library not available: {exc}")

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a support triage assistant. Follow instructions precisely.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_completion_tokens=3000,
        top_p=1,
    )
    content = response.choices[0].message.content or ""
    meta = {
        "provider": "groq",
        "model": model,
    }
    return content, meta


def _fallback_prediction(ticket_id: str, reason: str, config: Dict[str, Any]) -> Dict[str, Any]:
    route = config["routing_rules"].get("other", "manual_review_queue")
    return {
        "ticket_id": ticket_id,
        "category": "other",
        "priority": "normal",
        "reason": reason,
        "suggested_reply": "Thanks for contacting support. We are reviewing your request and will follow up soon.",
        "route_to": route,
        "confidence": 0.5,
    }


def _coerce_prediction(raw: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    allowed_categories = set(config.get("allowed_categories", []))
    allowed_priorities = set(config.get("allowed_priorities", []))
    routing_rules = config.get("routing_rules", {})
    reply_style = config.get("reply_style", {})

    ticket_id = str(raw.get("ticket_id", ""))
    category = str(raw.get("category", "other"))
    priority = str(raw.get("priority", "normal"))
    reason = str(raw.get("reason", ""))
    suggested_reply = str(raw.get("suggested_reply", ""))
    confidence = raw.get("confidence")

    if category not in allowed_categories:
        category = "other"
    if priority not in allowed_priorities:
        priority = "normal"
    route_to = routing_rules.get(category, routing_rules.get("other", "manual_review_queue"))

    max_words = int(reply_style.get("max_words", 80))
    if suggested_reply:
        suggested_reply = truncate_words(suggested_reply, max_words)
    else:
        suggested_reply = "Thanks for contacting support. We are reviewing your request and will follow up soon."

    if confidence is None:
        confidence = 0.7
    try:
        confidence = float(confidence)
    except Exception:
        confidence = 0.7

    if confidence < 0.0:
        confidence = 0.0
    if confidence > 1.0:
        confidence = 1.0

    if word_count(suggested_reply) > max_words:
        suggested_reply = truncate_words(suggested_reply, max_words)

    return {
        "ticket_id": ticket_id,
        "category": category,
        "priority": priority,
        "reason": reason,
        "suggested_reply": suggested_reply,
        "route_to": route_to,
        "confidence": confidence,
    }


def _call_with_retries(prompt: str, model: str, api_key: str, retries: int) -> Tuple[str, Dict[str, Any], float]:
    attempt = 0
    while True:
        try:
            start = time.perf_counter()
            content, meta = call_groq(prompt, model=model, api_key=api_key)
            latency = time.perf_counter() - start
            return content, meta, latency
        except Exception as exc:
            attempt += 1
            if attempt > retries:
                raise TriageError(str(exc))
            time.sleep(1.5 * attempt)


def predict_triage(
    normalized: List[Dict[str, Any]],
    config: Dict[str, Any],
    out_path: Path,
    llm_log_path: Path,
    error_log_path: Path,
    model: str,
    api_key: str,
    batch_size: int,
    retries: int,
) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    if batch_size <= 0:
        batch_size = len(normalized)

    for batch_index in range(0, len(normalized), batch_size):
        batch = normalized[batch_index : batch_index + batch_size]
        prompt = build_prompt(config, batch)
        prompt_hash = sha256_text(prompt)

        content, meta, latency = _call_with_retries(prompt, model=model, api_key=api_key, retries=retries)
        try:
            raw_items = _extract_json_array(content)
        except Exception as exc:
            raw_items = []
            errors.append(
                {
                    "ticket_id": None,
                    "error": f"failed to parse LLM output: {exc}",
                    "batch_index": batch_index,
                }
            )

        by_id = {str(item.get("ticket_id", "")): item for item in raw_items if isinstance(item, dict)}
        for ticket in batch:
            ticket_id = str(ticket.get("ticket_id", ""))
            if ticket_id in by_id:
                predictions.append(_coerce_prediction(by_id[ticket_id], config))
            else:
                errors.append(
                    {
                        "ticket_id": ticket_id,
                        "error": "missing ticket in LLM output",
                        "batch_index": batch_index,
                    }
                )
                predictions.append(_fallback_prediction(ticket_id, "missing ticket in LLM output", config))

        log_row = {
            "stage": "TRIAGE_PREDICTED",
            "timestamp": now_iso(),
            "provider": meta.get("provider", ""),
            "model": meta.get("model", ""),
            "prompt_hash": prompt_hash,
            "input_artifacts": ["normalized_tickets.json", "triage_config.json"],
            "output_artifact": out_path.name,
            "latency_seconds": round(latency, 4),
            "response_chars": len(content),
            "batch_index": batch_index,
            "batch_size": len(batch),
        }
        write_jsonl(llm_log_path, [log_row], append=True)

    write_json(out_path, predictions)
    if errors:
        write_jsonl(error_log_path, errors, append=True)
    return predictions
