#!/usr/bin/env python3
"""Evaluate a PI0 policy on ``put_blocks_into_boxes`` (T4).

* **Perception**: camera (pos+rot joint) + light DR sweeps → AUSC.
* **Understanding**: fixed seeds × {red, blue, green} instructions;
  env is rebuilt with matching ``special_cube`` so ``evaluate()`` scores
  the instructed colour. Confusion matrix uses ``*_in_left`` flags.

Sensor settings come from ``eval/configs/put_blocks_into_boxes.yaml``
(512×512, fov=1.0≈57°, shader=default) — matching the training replay.

Seeds must be pre-selected::

  python -m eval.select_eval_seeds --config eval/configs/put_blocks_into_boxes.yaml
  python -m eval.eval_put_blocks \\
      --policy-path outputs/pi0_put_blocks_mixed/checkpoints/030000/pretrained_model \\
      --seeds /tmp/put_blocks_dev.json \\
      --mode both
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import gymnasium as gym
import numpy as np
from tqdm import tqdm

import core.env  # noqa: F401
from mani_skill.utils.wrappers import CPUGymWrapper
from utils.eval_common import (
    build_base_env_kwargs,
    build_perception_profiles,
    dump_json,
    load_pi0_policy,
    load_seed_list,
    load_task_config,
    resolve_path,
    run_episode,
    summarise_perception,
)

DEFAULT_CONFIG = "eval/configs/put_blocks_into_boxes.yaml"
COLORS = ("red", "blue", "green")


def _make_env(env_kwargs: dict):
    env = gym.make("put_blocks_into_boxes", **env_kwargs)
    return CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)


def _cube_in_left(extras: dict) -> Optional[str]:
    """Pick which colour (if any) ended in the left box; prefer a unique hit."""
    ev = extras.get("evaluate") or {}
    hits = [c for c in COLORS if _as_bool(ev.get(f"{c}_in_left"))]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "multi:" + ",".join(hits)
    return "none"


def _as_bool(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (list, tuple)):
        return bool(v[0]) if v else False
    try:
        return bool(v)
    except Exception:
        return False


def run_perception(
    cfg: dict,
    policy,
    preprocessor,
    postprocessor,
    seeds: List[int],
    *,
    record_dir: Path,
) -> dict:
    perc = cfg.get("perception") or {}
    variant = perc.get("default_variant", "green")
    instruction = (cfg.get("variants") or {})[variant]
    max_steps = int(cfg.get("max_episode_steps", 800))

    base_kw = build_base_env_kwargs(cfg, extra={"special_cube": variant})
    profiles = build_perception_profiles(base_kw, cfg)
    n = min(int(perc.get("n_episodes", len(seeds))), len(seeds))
    use_seeds = seeds[:n]

    profile_results: Dict[str, Any] = {}
    for name, kw in profiles.items():
        successes, stage_ids, episodes = [], [], []
        env = _make_env(kw)
        try:
            pbar = tqdm(use_seeds, desc=f"perception[{name}]")
            for seed in pbar:
                ep = run_episode(
                    env, policy, preprocessor, postprocessor,
                    task=instruction, seed=seed, max_steps=max_steps,
                )
                successes.append(ep["success"])
                sid = ep["extras"].get("stage_id", 0)
                stage_ids.append(sid)
                episodes.append({
                    "seed": seed, "success": ep["success"],
                    "episode_length": ep["episode_length"],
                    "stage_id": sid,
                })
                pbar.set_postfix(sr=f"{np.mean(successes):.3f}")
        finally:
            env.close()
        profile_results[name] = {
            "episodes": len(successes),
            "successes": int(np.sum(successes)),
            "success_rate": float(np.mean(successes)) if successes else float("nan"),
            "mean_stage_id": float(np.mean(stage_ids)) if stage_ids else float("nan"),
            "details": episodes,
        }
        print(f"  {name}: sr={profile_results[name]['success_rate']:.3f} "
              f"mean_stage={profile_results[name]['mean_stage_id']:.2f}")

    summary = summarise_perception(profile_results)
    summary["meta"] = {
        "task_id": cfg.get("task_id"),
        "mode": "perception",
        "variant": variant,
        "instruction": instruction,
        "n_seeds": n,
        "seeds": use_seeds,
    }
    out = record_dir / "perception_summary.json"
    dump_json(out, summary)
    print(f"saved {out}")
    for k in ("ausc_camera", "ausc_light", "ausc_mean"):
        if k in summary:
            print(f"{k}={summary[k]['value']:.4f}")
    return summary


def run_understanding(
    cfg: dict,
    policy,
    preprocessor,
    postprocessor,
    seeds: List[int],
    *,
    record_dir: Path,
) -> dict:
    variants: Dict[str, str] = dict(cfg.get("variants") or {})
    max_steps = int(cfg.get("max_episode_steps", 800))

    confusion: Dict[str, Dict[str, int]] = {v: defaultdict(int) for v in variants}
    per_variant_success: Dict[str, List[bool]] = {v: [] for v in variants}
    stage_hist: Dict[int, int] = defaultdict(int)
    episodes: List[dict] = []

    envs = {
        variant: _make_env(build_base_env_kwargs(cfg, extra={"special_cube": variant}))
        for variant in variants
    }
    try:
        for seed in tqdm(seeds, desc="understanding"):
            for variant, instruction in variants.items():
                ep = run_episode(
                    envs[variant], policy, preprocessor, postprocessor,
                    task=instruction, seed=seed, max_steps=max_steps,
                )
                left = _cube_in_left(ep["extras"])
                confusion[variant][left] += 1
                per_variant_success[variant].append(ep["success"])
                sid = int(ep["extras"].get("stage_id", 0))
                stage_hist[sid] += 1
                episodes.append({
                    "seed": seed,
                    "instructed": variant,
                    "instruction": instruction,
                    "success": ep["success"],
                    "cube_in_left": left,
                    "stage_id": sid,
                    "episode_length": ep["episode_length"],
                })
    finally:
        for e in envs.values():
            try:
                e.close()
            except Exception:
                pass

    n_total = len(episodes)
    n_ok = sum(1 for e in episodes if e["success"])
    # Behavior: fraction of stages completed (stage_id / 3), averaged.
    stage_fracs = [e["stage_id"] / 3.0 for e in episodes]
    summary = {
        "meta": {
            "task_id": cfg.get("task_id"),
            "mode": "understanding",
            "n_seeds": len(seeds),
            "seeds": seeds,
            "variants": list(variants.keys()),
        },
        "success_rate": float(n_ok / n_total) if n_total else float("nan"),
        "n_episodes": n_total,
        "n_successes": n_ok,
        "per_variant_success_rate": {
            v: float(np.mean(s)) if s else float("nan")
            for v, s in per_variant_success.items()
        },
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "stage_histogram": {str(k): v for k, v in sorted(stage_hist.items())},
        "behavior": {
            "final_success_rate": float(n_ok / n_total) if n_total else float("nan"),
            "mean_stage_fraction": float(np.mean(stage_fracs)) if stage_fracs else float("nan"),
            "note": "multi-stage: Behavior reports final SR + mean(stage_id/3)",
        },
        "episodes": episodes,
    }
    out = record_dir / "understanding_summary.json"
    dump_json(out, summary)
    print(f"saved {out}")
    print(f"Understanding SR={summary['success_rate']:.3f}")
    print(f"per-variant: {summary['per_variant_success_rate']}")
    print(f"confusion: {summary['confusion']}")
    print(f"stage_hist: {summary['stage_histogram']}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--policy-path", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--mode", choices=["perception", "understanding", "both"], default="both")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--record-dir", default="eval_runs/put_blocks_into_boxes")
    p.add_argument("--n-seeds", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_task_config(args.config)
    seeds = load_seed_list(args.seeds)
    if args.n_seeds is not None:
        seeds = seeds[: args.n_seeds]
    record_dir = resolve_path(args.record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)

    print(f"config={args.config}")
    print(f"policy={args.policy_path}")
    print(f"seeds={len(seeds)} from {args.seeds}")
    print(f"sensor={cfg.get('sensor')}")

    policy, preprocessor, postprocessor = load_pi0_policy(
        args.policy_path, device=args.device, tokenizer_path=args.tokenizer_path,
    )

    if args.mode in ("perception", "both"):
        run_perception(
            cfg, policy, preprocessor, postprocessor, seeds, record_dir=record_dir,
        )
    if args.mode in ("understanding", "both"):
        run_understanding(
            cfg, policy, preprocessor, postprocessor, seeds, record_dir=record_dir,
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
