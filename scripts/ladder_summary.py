"""Summarise Stage A ladder runs (DECISIONS #15) from their metrics.jsonl.

For each run directory under ``runs/ladder``: the LR-selection criterion
(mean train CE over the last 1 000 steps, progressive validation on unseen
text), the final protocol PPL and the number of logged steps.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def summarise(run_dir: Path, last: int) -> dict[str, object] | None:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    losses: dict[int, float] = {}
    protocol_ppl = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("loss") is not None:
            losses[int(row["step"])] = float(row["loss"])
        if row.get("protocol_ppl") is not None:
            protocol_ppl = float(row["protocol_ppl"])
    if not losses:
        return None
    last_step = max(losses)
    window = [v for s, v in losses.items() if s > last_step - last]
    ce = sum(window) / len(window)
    return {
        "run": run_dir.name, "last_step": last_step,
        "train_ce_last": ce, "train_ppl_last": math.exp(ce),
        "protocol_ppl": protocol_ppl,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs/ladder")
    p.add_argument("--prefix", default="")
    p.add_argument("--last", type=int, default=1000, help="steps in the CE window")
    args = p.parse_args()
    rows = [
        r for d in sorted(Path(args.runs).iterdir())
        if d.is_dir() and d.name.startswith(args.prefix)
        and (r := summarise(d, args.last)) is not None
    ]
    print(f"{'run':32s} {'step':>6s} {'train CE':>9s} {'protocol PPL':>13s}")
    for r in rows:
        ppl = f"{r['protocol_ppl']:.3f}" if r["protocol_ppl"] is not None else "-"
        print(f"{r['run']:32s} {r['last_step']:6d} {r['train_ce_last']:9.4f} {ppl:>13s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
