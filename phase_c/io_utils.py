# phase_c/io_utils.py
from __future__ import annotations
import json, csv, hashlib, os, time
from pathlib import Path
from typing import Dict, Any, Tuple

def ensure_dir(p: str | Path) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def write_json(path: str | Path, obj: Dict[str, Any]) -> str:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return sha256_file(path)

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 16)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def append_index(index_path: str | Path,
                 row: Dict[str, Any],
                 field_order=("run_id","backend","config_path","config_sha256","metrics_path","metrics_sha256",
                              "N","r","dt","omega","kappa","chi","noise","noise_p","shots","label")) -> None:
    exists = Path(index_path).exists()
    with open(index_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(field_order))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def extract_execution_ms(result) -> int | None:
    """
    Try multiple places where Braket simulators report execution duration.
    Returns milliseconds or None if unavailable.
    """
    try:
        meta = result.additional_metadata  # SDK convenience
    except Exception:
        meta = None
    # Known spots:
    for path in [
        ("simulatorMetadata","executionDuration"),   # common
        ("taskMetadata","executionDuration"),
        ("responseMetadata","executionDurationMs"),
    ]:
        cur = meta
        try:
            for k in path:
                cur = getattr(cur, k) if hasattr(cur, k) else cur.get(k)  # dict or attr
            if cur is not None:
                return int(cur)
        except Exception:
            pass
    return None

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")
