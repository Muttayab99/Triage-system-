from pathlib import Path
from typing import Any, Dict, List

from src.utils import write_json


def normalize_ticket(ticket: Dict[str, Any]) -> Dict[str, Any]:
    subject = str(ticket.get("subject", ""))
    message = str(ticket.get("message", ""))
    text_for_model = f"Subject: {subject}\n\nMessage: {message}"
    return {
        "ticket_id": str(ticket.get("ticket_id", "")),
        "subject": subject,
        "message": message,
        "channel": str(ticket.get("channel", "")),
        "created_at": str(ticket.get("created_at", "")),
        "text_for_model": text_for_model,
        "char_count": len(text_for_model),
    }


def normalize_tickets(tickets: List[Dict[str, Any]], out_path: Path) -> List[Dict[str, Any]]:
    normalized = [normalize_ticket(t) for t in tickets]
    write_json(out_path, normalized)
    return normalized
