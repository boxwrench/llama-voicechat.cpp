#!/usr/bin/env python3

import csv
import json
import math
import statistics
from pathlib import Path


CASES = [
    ("VC01", "primary"),
    ("VC02", "VC02-conversation"),
    ("VC03", "VC03-long"),
    ("VC04", "VC04-noisy"),
    ("VC05", "VC05-pause"),
    ("VC06", "VC06-tool"),
]
METRICS = ["total_ms", "first_text_ms", "audio_ready_ms", "perception_ms", "timeline_ms", "drain_ms", "total_s2s_rtf"]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    baseline = repo / "research/amd-voicechat/baselines/R9700-Q8-M1"
    generated = baseline / "generated"
    all_rows = []
    summaries = []

    for case, directory in CASES:
        with (generated / directory / "raw-runs.csv").open(newline="", encoding="utf-8") as src:
            rows = list(csv.DictReader(src))
        for row in rows:
            row = {"case": case, **row}
            row.setdefault("tool_calls", "")
            all_rows.append(row)

        measured = [row for row in rows if row["warmup"] == "0"]
        texts = {row["text"] for row in measured}
        wavs = {row["output_sha256"] for row in measured}
        calls = sum(len(json.loads(row.get("tool_calls") or "[]")) for row in measured)
        for metric in METRICS:
            values = [float(row[metric]) for row in measured if row.get(metric) not in (None, "")]
            summaries.append({
                "case": case,
                "metric": metric,
                "count": len(values),
                "mean": f"{statistics.fmean(values):.3f}",
                "p50": f"{percentile(values, 0.50):.3f}",
                "p95": f"{percentile(values, 0.95):.3f}",
                "min": f"{min(values):.3f}",
                "max": f"{max(values):.3f}",
                "text_variants": len(texts),
                "wav_variants": len(wavs),
                "tool_calls": calls,
            })

    fields = list(all_rows[0])
    with (baseline / "raw-runs.csv").open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    fields = list(summaries[0])
    with (baseline / "summary.csv").open("w", newline="", encoding="utf-8") as dst:
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
