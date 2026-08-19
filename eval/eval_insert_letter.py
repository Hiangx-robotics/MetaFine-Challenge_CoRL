#!/usr/bin/env python3
"""Evaluate a PI0 policy on ``insert_letter`` (T5).

* **Perception**: camera (pos+rot joint) + light DR sweeps → AUSC.
* **Understanding**: fixed seeds × {C, o, R, L} instructions;
  env is rebuilt with matching ``target_letter``. Confusion uses
  ``inserted_*`` flags from ``evaluate()``.

Sensor settings come from ``eval/configs/insert_letter.yaml``
(512×512, fov≈70°, shader=default) — matching the training replay.

Seeds must be pre-selected::

  python -m eval.select_eval_seeds --config eval/configs/insert_letter.yaml
  python -m eval.eval_insert_letter \\
      --policy-path outputs/pi0_insert_letter_mixed/checkpoints/030000/pretrained_model \\
      --seeds /tmp/insert_letter_dev.json \\
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
from core.letter_glyphs import LETTERS
from mani_skill.utils.wrappers import CPUGymWrapper
from utils.eval_common import (
    build_base_env_kwargs,
    build_perception_profiles,
    dump_json,
    load_pi0_policy,
    load_seed_list,
    load_task_config,
    obs_to_frame,
    resolve_path,
    run_episode,
    save_rgb_video,
    summarise_perception,
)

DEFAULT_CONFIG = "eval/configs/insert_letter.yaml"


def _make_env(env_kwargs: dict):
    env = gym.make("insert_letter", **env_kwargs)
    return CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)


def _as_bool(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, (list, tuple)):
        return bool(v[0]) if v else False
    if hasattr(v, "item"):
        try:
            return bool(v.item())
        except Exception:
            pass
    try:
        return bool(v)
    except Exception:
        return False


def _frame_probe():
    """Collect policy-view frames for optional mp4 export."""
    frames: List[np.ndarray] = []

    def on_step(_t: int, obs: Any, _info: Any) -> None:
        try:
            frames.append(obs_to_frame(obs))
        except Exception:
            pass

    return on_step, frames


def _video_name(*, seed: int, variant: str, success: bool, inserted: str, length: int) -> str:
    return (
        f"seed{int(seed):06d}_{variant}_succ{int(bool(success))}"
        f"_in-{inserted}_len{int(length):03d}.mp4"
    )


def _inserted_letter(extras: dict) -> str:
    ev = extras.get("evaluate") or {}
    hits = [L for L in LETTERS if _as_bool(ev.get(f"inserted_{L}"))]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "multi:" + ",".join(hits)
    return "none"


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return _as_float(v[0]) if v else None
    try:
        return float(v)
    except Exception:
        return None


def _ev_detail(extras: dict) -> dict:
    """Per-episode failure attribution.

    ``success = in own slot & released & settled``, so a bare pass/fail cannot
    say whether a miss was translation, yaw, or a policy that inserted the peg
    but never let go. Carry the terms so the summary can answer that.
    """
    ev = extras.get("evaluate") or {}
    return {
        "pos_diff_norm": _as_float(ev.get("pos_diff_norm")),
        "rot_diff": _as_float(ev.get("rot_diff")),
        "pos_correct": _as_bool(ev.get("pos_correct")),
        "rot_correct": _as_bool(ev.get("rot_correct")),
        "in_slot": _as_bool(ev.get("in_slot")),
        "not_grasped": _as_bool(ev.get("not_grasped")),
        "is_static": _as_bool(ev.get("is_static")),
    }


def _failure_modes(episodes: List[dict], *, near_slot: float = 0.10) -> dict:
    """Bucket failures into mutually exclusive, exhaustive categories.

    ``in_slot`` alone is not evidence of insertion: it only tests peg z, and a
    peg still lying on the table clears that threshold too. So the lateral
    error drives the classification, and ``near_slot`` separates "approached
    the right slot but never seated" from "never brought the peg over".
    """
    fails = [e for e in episodes if not e["success"]]
    counts = {
        "untouched_or_far": 0,   # peg never brought over its slot
        "hover_not_seated": 0,   # over the slot laterally, never dropped in
        "yaw_off": 0,            # seated laterally, wrong orientation
        "seated_but_held": 0,    # inserted but gripper never released
        "seated_but_moving": 0,  # inserted and released but still settling
        "other": 0,
    }
    for e in fails:
        pos = e.get("pos_diff_norm")
        if e.get("pos_correct") and e.get("rot_correct") and e.get("in_slot"):
            if not e.get("not_grasped"):
                counts["seated_but_held"] += 1
            elif not e.get("is_static"):
                counts["seated_but_moving"] += 1
            else:
                counts["other"] += 1
        elif e.get("pos_correct") and not e.get("rot_correct"):
            counts["yaw_off"] += 1
        elif pos is not None and pos <= near_slot:
            counts["hover_not_seated"] += 1
        elif pos is not None:
            counts["untouched_or_far"] += 1
        else:
            counts["other"] += 1
    counts["n_failures"] = len(fails)
    return counts


def run_perception(
    cfg: dict,
    policy,
    preprocessor,
    postprocessor,
    seeds: List[int],
    *,
    record_dir: Path,
    save_video: bool = False,
    video_fps: int = 30,
    max_videos: int = 6,
) -> dict:
    perc = cfg.get("perception") or {}
    variant = perc.get("default_variant", "C")
    instruction = (cfg.get("variants") or {})[variant]
    max_steps = int(cfg.get("max_episode_steps", 400))

    base_kw = build_base_env_kwargs(cfg, extra={"target_letter": variant})
    profiles = build_perception_profiles(base_kw, cfg)
    n = min(int(perc.get("n_episodes", len(seeds))), len(seeds))
    use_seeds = seeds[:n]

    profile_results: Dict[str, Any] = {}
    for name, kw in profiles.items():
        successes, episodes = [], []
        want_video = bool(save_video)
        video_dir = record_dir / "videos" / f"perception_{name}"
        n_saved = 0
        env = _make_env(kw)
        try:
            pbar = tqdm(use_seeds, desc=f"perception[{name}]")
            for seed in pbar:
                grab = want_video and n_saved < int(max_videos)
                on_step, frames = _frame_probe() if grab else (None, [])
                ep = run_episode(
                    env, policy, preprocessor, postprocessor,
                    task=instruction, seed=seed, max_steps=max_steps,
                    on_step=on_step,
                )
                successes.append(ep["success"])
                inserted = _inserted_letter(ep["extras"])
                detail = {
                    "seed": seed,
                    "success": ep["success"],
                    "episode_length": ep["episode_length"],
                    "inserted": inserted,
                    **_ev_detail(ep["extras"]),
                }
                if grab and frames:
                    vpath = video_dir / _video_name(
                        seed=seed, variant=variant, success=ep["success"],
                        inserted=inserted, length=ep["episode_length"],
                    )
                    save_rgb_video(frames, vpath, fps=video_fps)
                    detail["video"] = str(vpath.relative_to(record_dir))
                    n_saved += 1
                episodes.append(detail)
                pbar.set_postfix(sr=f"{np.mean(successes):.3f}")
        finally:
            env.close()
        profile_results[name] = {
            "episodes": len(successes),
            "successes": int(np.sum(successes)),
            "success_rate": float(np.mean(successes)) if successes else float("nan"),
            "failure_modes": _failure_modes(episodes),
            "details": episodes,
        }
        print(f"  {name}: sr={profile_results[name]['success_rate']:.3f}")

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
    save_video: bool = False,
    video_fps: int = 30,
    max_videos: int = 6,
) -> dict:
    variants: Dict[str, str] = dict(cfg.get("variants") or {})
    max_steps = int(cfg.get("max_episode_steps", 400))
    n_saved: Dict[str, int] = {v: 0 for v in variants}

    confusion: Dict[str, Dict[str, int]] = {v: defaultdict(int) for v in variants}
    per_variant_success: Dict[str, List[bool]] = {v: [] for v in variants}
    episodes: List[dict] = []

    envs = {
        variant: _make_env(build_base_env_kwargs(cfg, extra={"target_letter": variant}))
        for variant in variants
    }
    try:
        for seed in tqdm(seeds, desc="understanding"):
            for variant, instruction in variants.items():
                grab = bool(save_video) and n_saved[variant] < int(max_videos)
                on_step, frames = _frame_probe() if grab else (None, [])
                ep = run_episode(
                    envs[variant], policy, preprocessor, postprocessor,
                    task=instruction, seed=seed, max_steps=max_steps,
                    on_step=on_step,
                )
                inserted = _inserted_letter(ep["extras"])
                confusion[variant][inserted] += 1
                per_variant_success[variant].append(ep["success"])
                detail = {
                    "seed": seed,
                    "instructed": variant,
                    "instruction": instruction,
                    "success": ep["success"],
                    "inserted": inserted,
                    "episode_length": ep["episode_length"],
                    **_ev_detail(ep["extras"]),
                }
                if grab and frames:
                    vpath = record_dir / "videos" / "understanding" / variant / _video_name(
                        seed=seed, variant=variant, success=ep["success"],
                        inserted=inserted, length=ep["episode_length"],
                    )
                    save_rgb_video(frames, vpath, fps=video_fps)
                    detail["video"] = str(vpath.relative_to(record_dir))
                    n_saved[variant] += 1
                episodes.append(detail)
    finally:
        for e in envs.values():
            try:
                e.close()
            except Exception:
                pass

    n_total = len(episodes)
    n_ok = sum(1 for e in episodes if e["success"])
    failure_modes = _failure_modes(episodes)
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
        "failure_modes": failure_modes,
        "behavior": {
            "final_success_rate": float(n_ok / n_total) if n_total else float("nan"),
            "note": "single-stage insert: Behavior ≡ final success",
        },
        "episodes": episodes,
    }
    out = record_dir / "understanding_summary.json"
    dump_json(out, summary)
    print(f"saved {out}")
    print(f"Understanding SR={summary['success_rate']:.3f}")
    print(f"per-variant: {summary['per_variant_success_rate']}")
    print(f"confusion: {summary['confusion']}")
    print(f"failure_modes: {summary['failure_modes']}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--policy-path", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--mode", choices=["perception", "understanding", "both"], default="both")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--record-dir", default="eval_runs/insert_letter")
    p.add_argument("--n-seeds", type=int, default=None)
    p.add_argument("--save-video", action="store_true",
                   help="Write side-by-side base+hand camera mp4s (policy view).")
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--max-videos", type=int, default=6,
                   help="Cap per bucket (clean profile / each instruction variant).")
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

    video_kw = dict(
        save_video=args.save_video,
        video_fps=args.video_fps,
        max_videos=args.max_videos,
    )
    if args.mode in ("perception", "both"):
        run_perception(
            cfg, policy, preprocessor, postprocessor, seeds,
            record_dir=record_dir, **video_kw,
        )
    if args.mode in ("understanding", "both"):
        run_understanding(
            cfg, policy, preprocessor, postprocessor, seeds,
            record_dir=record_dir, **video_kw,
        )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
