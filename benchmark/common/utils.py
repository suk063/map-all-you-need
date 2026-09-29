"""Small helpers for local experiment outputs."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def run_directory(output, env_id, mode, seed):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(output or f"runs/rl/{env_id}/{mode}/seed{seed}-{stamp}").resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def log_metrics(path, step, metrics):
    """Long-form CSV also handles task-specific and training/evaluation metrics."""
    exists = Path(path).exists()
    with open(path, "a", newline="") as stream:
        writer = csv.writer(stream)
        if not exists:
            writer.writerow(("steps", "metric", "value"))
        writer.writerows((int(step), key, float(value)) for key, value in sorted(metrics.items()))
    print(json.dumps({"steps": int(step), **{k: float(v) for k, v in metrics.items()}}), flush=True)
