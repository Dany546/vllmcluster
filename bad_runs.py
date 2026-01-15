import csv
import os
from datetime import datetime
from typing import Dict, Any


DEFAULT_BAD_RUNS_PATH = os.path.join(
    "/globalscratch/ucl/irec/darimez/dino/proj/", "bad_runs.csv"
)


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def append_bad_run(
    run_id: str,
    algo: str,
    requested_n_components: int,
    observed_n_features: int,
    hyperparams: Dict[str, Any],
    path: str = DEFAULT_BAD_RUNS_PATH,
):
    """Append a bad run entry to CSV (idempotent append).

    Columns: timestamp, run_id, algo, requested_n_components, observed_n_features, hyperparams_json
    """
    ensure_dir(path)
    row = [
        datetime.utcnow().isoformat(),
        run_id,
        algo,
        int(requested_n_components) if requested_n_components is not None else "",
        int(observed_n_features),
        json_safe(hyperparams),
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow([
                "timestamp",
                "run_id",
                "algo",
                "requested_n_components",
                "observed_n_features",
                "hyperparams",
            ])
        writer.writerow(row)


def json_safe(obj: Dict[str, Any]) -> str:
    try:
        import json

        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


def load_bad_runs(path: str = DEFAULT_BAD_RUNS_PATH):
    if not os.path.exists(path):
        return set()
    ids = set()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                ids.add(r.get("run_id"))
    except Exception:
        return set()
    return ids


def is_bad_run(run_id: str, path: str = DEFAULT_BAD_RUNS_PATH) -> bool:
    return run_id in load_bad_runs(path)
