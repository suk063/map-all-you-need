import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Small MLPs and HDF5 samples otherwise suffer on large CPU machines.
    torch.set_num_threads(4)


def run_directory(output, algorithm, env_id, mode, seed):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(output or f"runs/{algorithm}/{env_id}/{mode}/seed{seed}-{stamp}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def log_row(path, row):
    exists = Path(path).exists()
    with open(path, "a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row), flush=True)
