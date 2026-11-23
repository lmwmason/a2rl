"""Training metric readers."""
import json
import os


def read_metrics(run):
    try:
        with (run / "metrics.jsonl").open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 350_000))
            if size > 350_000:
                stream.readline()
            rows = []
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows
    except OSError:
        return []


def finish_time_history(metrics, speed):
    history = []
    for row in metrics[-60:]:
        index = row.get("steps" if speed else "round")
        if index is None:
            continue
        history.append({
            "at": index,
            "standard_s": row.get("official_start_seconds") if speed else None,
            "comparison_s": row.get("official_mean_seconds" if speed else "best_two_lap_s"),
        })
    return history
