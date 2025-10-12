# services/log_sink.py
from __future__ import annotations

import json
import os
import time
import threading
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from scripts import config

_LOG_LOCK = threading.Lock()

# Ensure logs dir exists
try:
    (config.EXPL_DIR / "logs").mkdir(parents=True, exist_ok=True)
except Exception:
    pass

_TOOL_LOG = (config.EXPL_DIR / "logs" / "tool-metrics.jsonl")
_USER_LOG = (config.EXPL_DIR / "logs" / "user-metrics.jsonl")


def _iso_utc(ts: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _stable_row_id(row: Dict[str, Any]) -> str:
    """
    Deterministic hash of the row *excluding* any existing 'row_id' field.
    We keep 'ts'/'ts_utc' in the hash so two otherwise identical rows written
    at different times still get different IDs.
    """
    try:
        payload = dict(row)
        payload.pop("row_id", None)
        s = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
    except Exception:
        # ultra-safe fallback
        return hashlib.sha256(repr(row).encode("utf-8")).hexdigest()


def _append_jsonl(path: str, row: Dict[str, Any]) -> bool:
    # stamp ts, ts_utc, row_id if missing
    row = dict(row)
    row.setdefault("ts", int(time.time()))
    row.setdefault("ts_utc", _iso_utc(float(row["ts"])))
    row.setdefault("row_id", _stable_row_id(row))

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(row, ensure_ascii=False)
        with _LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return True
    except Exception:
        return False


def _read_jsonl_all(path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file into a list of dict rows.
    Backfills 'row_id' for older rows that don't have it yet.
    """
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with _LOG_LOCK:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except Exception:
                    continue
                if "row_id" not in row:
                    row["row_id"] = _stable_row_id(row)
                if "ts_utc" not in row:
                    try:
                        row["ts_utc"] = _iso_utc(float(row.get("ts", time.time())))
                    except Exception:
                        row["ts_utc"] = _iso_utc()
                rows.append(row)
    return rows


def latest_tool_row_for(canonical_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the most recent tool log row for a given canonical_id, or None.
    """
    rows = _read_jsonl_all(str(_TOOL_LOG))
    rows = [r for r in rows if str(r.get("canonical_id", "")) == str(canonical_id)]
    if not rows:
        return None
    rows.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return rows[0]


def append_tool_metrics(payload: Dict[str, Any]) -> bool:
    """
    Expected keys (flexible): canonical_id (required), form_title?, bucket?, metrics? (dict),
    source? ("analysis" | "seed" | "training" | "funsd"), note?
    """
    if not isinstance(payload, dict) or not payload.get("canonical_id"):
        return False
    row = {
        "ts": int(time.time()),
        "ts_utc": _iso_utc(),
        "source": (payload.get("source") or "analysis"),
        "canonical_id": payload["canonical_id"],
        "form_title": payload.get("form_title"),
        "bucket": payload.get("bucket"),
        "metrics": payload.get("metrics") or {},
        "note": payload.get("note"),
    }
    # row_id stamped in _append_jsonl
    return _append_jsonl(str(_TOOL_LOG), row)


def append_user_metrics(payload: Dict[str, Any]) -> bool:
    """
    Expected keys: user_id (caps string), canonical_id, method ("intelliform"|"vanilla"|"manual"),
    started_at (epoch ms or iso), finished_at, duration_ms
    """
    if not isinstance(payload, dict) or not payload.get("canonical_id") or not payload.get("user_id"):
        return False
    row = {
        "ts": int(time.time()),
        "ts_utc": _iso_utc(),
        "user_id": payload.get("user_id"),
        "canonical_id": payload.get("canonical_id"),
        "method": payload.get("method") or "intelliform",
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "duration_ms": payload.get("duration_ms"),
        "meta": payload.get("meta") or {},
    }
    # row_id stamped in _append_jsonl
    return _append_jsonl(str(_USER_LOG), row)
