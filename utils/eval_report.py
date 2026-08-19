#!/usr/bin/env python3
"""Aggregate Perception / Understanding / Behavior into one MetaFine report.

Reads the per-mode JSON files written by ``eval/eval_*.py`` and prints a
compact table + ``metafine_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from utils.eval_common import dump_json, resolve_path


def _load(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = resolve_path(path)
    if not p.exists():
        print(f"[warn] missing: {p}", file=sys.stderr)
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def build_report(
    *,
    task_id: str,
    perception: Optional[dict],
    understanding: Optional[dict],
) -> dict:
    report: dict[str, Any] = {"task_id": task_id, "metrics": {}}

    if perception:
        report["metrics"]["perception"] = {
            "ausc_camera": (perception.get("ausc_camera") or {}).get("value"),
            "ausc_light": (perception.get("ausc_light") or {}).get("value"),
            "ausc_mean": (perception.get("ausc_mean") or {}).get("value"),
            "clean_success_rate": (perception.get("clean") or {}).get("success_rate"),
            "curves": {
                "camera": (perception.get("ausc_camera") or {}).get("curve"),
                "light": (perception.get("ausc_light") or {}).get("curve"),
            },
        }

    if understanding:
        behavior = understanding.get("behavior") or {}
        report["metrics"]["understanding"] = {
            "success_rate": understanding.get("success_rate"),
            "per_variant_success_rate": understanding.get("per_variant_success_rate"),
            "confusion": understanding.get("confusion"),
        }
        report["metrics"]["behavior"] = {
            "success_rate": behavior.get("success_rate", behavior.get("final_success_rate")),
            "mean_stage_fraction": behavior.get("mean_stage_fraction"),
            "note": behavior.get("note"),
        }

    return report


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def print_report(report: dict) -> None:
    m = report.get("metrics") or {}
    print("=" * 56)
    print(f"MetaFine report — {report.get('task_id')}")
    print("=" * 56)
    perc = m.get("perception") or {}
    und = m.get("understanding") or {}
    beh = m.get("behavior") or {}
    print(f"Perception  AUSC(mean)   : {_fmt(perc.get('ausc_mean'))}")
    print(f"            AUSC(camera) : {_fmt(perc.get('ausc_camera'))}")
    print(f"            AUSC(light)  : {_fmt(perc.get('ausc_light'))}")
    print(f"            clean SR     : {_fmt(perc.get('clean_success_rate'))}")
    print(f"Understanding SR         : {_fmt(und.get('success_rate'))}")
    if und.get("per_variant_success_rate"):
        print(f"            per-variant  : {und['per_variant_success_rate']}")
    if und.get("confusion"):
        print(f"            confusion    : {und['confusion']}")
    print(f"Behavior    SR           : {_fmt(beh.get('success_rate'))}")
    if beh.get("mean_stage_fraction") is not None:
        print(f"            mean stage%  : {_fmt(beh.get('mean_stage_fraction'))}")
    print("=" * 56)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task-id", required=True)
    p.add_argument("--perception", default=None, help="perception_summary.json")
    p.add_argument("--understanding", default=None, help="understanding_summary.json")
    p.add_argument("--out", default=None, help="metafine_report.json path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        task_id=args.task_id,
        perception=_load(args.perception),
        understanding=_load(args.understanding),
    )
    print_report(report)
    out = args.out or f"eval_runs/{args.task_id}/metafine_report.json"
    dump_json(out, report)
    print(f"wrote {resolve_path(out)}")


if __name__ == "__main__":
    main()
