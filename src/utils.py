import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(data: Any) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=True)
    return sha256_text(payload)


def word_count(text: str) -> int:
    return len(text.split())


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def load_env_file(env_path: Path) -> Dict[str, str]:
    if not env_path.exists():
        return {}
    env: Dict[str, str] = {}
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    return env


def get_env(key: str, fallbacks: Optional[List[str]] = None, env_file: Optional[Path] = None) -> Optional[str]:
    if key in os.environ:
        return os.environ[key]
    if fallbacks:
        for name in fallbacks:
            if name in os.environ:
                return os.environ[name]
    if env_file:
        env = load_env_file(env_file)
        if key in env:
            return env[key]
        if fallbacks:
            for name in fallbacks:
                if name in env:
                    return env[name]
    return None
