#!/usr/bin/env python3
"""Evaluate a PI0 policy on ``toggle_switch_table`` (T3).

Two modes (can be combined in one run):

* **Perception** (``--mode perception``): sweep camera (pos+rot joint) and
  light DR levels; report per-level success rates + AUSC.
* **Understanding** (``--mode understanding``): fix each seed, run every
  instruction variant (red / blue); report success rate + confusion matrix
  based on which slider actually moved (``red_toggled`` / ``blue_toggled``).

Sensor settings are read from ``eval/configs/toggle_switch_table.yaml`` so they
stay aligned with the training replay metadata (512×512, fov≈70°).

Seeds must be pre-selected and saved (RoboTwin-style)::

  python -m eval.select_eval_seeds --config eval/configs/toggle_switch_table.yaml
  python -m eval.eval_toggle_switch \\
      --policy-path outputs/pi0_toggle_mixed/checkpoints/030000/pretrained_model \\
      --seeds /tmp/toggle_dev.json \\
      --mode both
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    save_rgb_video,
    summarise_perception,
)

DEFAULT_CONFIG = "eval/configs/toggle_switch_table.yaml"


def _variant_extra(cfg: dict, variant: str) -> dict:
    """Map instruction variant → gym.make kwargs."""
    key = cfg.get("variant_env_kwarg") or "target_switch"
    extra: dict[str, Any] = {str(key): variant}
    if cfg.get("object_name") is not None:
        extra["object_name"] = cfg["object_name"]
    if cfg.get("part_name") is not None:
        extra["part_name"] = cfg["part_name"]
    return extra


def _make_env(env_kwargs: dict):
    env = gym.make("toggle_switch_table", **env_kwargs)
    return CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)


def _toggled_color(info: dict) -> str:
    """Which coloured slider moved past the toggle threshold (else ``none``)."""
    red = info.get("red_toggled")
    blue = info.get("blue_toggled")
    if hasattr(red, "item"):
        red = bool(red.item()) if getattr(red, "numel", lambda: 1)() == 1 else bool(red)
    if hasattr(blue, "item"):
        blue = bool(blue.item()) if getattr(blue, "numel", lambda: 1)() == 1 else bool(blue)
    red = bool(red)
    blue = bool(blue)
    if red and blue:
        return "both"
    if red:
        return "red"
    if blue:
        return "blue"
    return "none"


def _episode_probe(
    env,
    *,
    collect_frames: bool = False,
) -> Tuple[Callable[[int, dict, Any], None], Callable[[], dict]]:
    """Capture RGB frames + final evaluate() toggles / contact."""
    from utils.eval_common import obs_to_frame

    frames: List[np.ndarray] = []
    last_info: dict = {}
    had_contact = False

    def on_step(step_i: int, info: dict, obs: Any) -> None:
        nonlocal last_info, had_contact
        last_info = dict(info) if info else {}
        if info and info.get("had_contact"):
            hc = info["had_contact"]
            if hasattr(hc, "item"):
                hc = bool(hc.item())
            had_contact = had_contact or bool(hc)
        if collect_frames:
            fr = obs_to_frame(obs)
            if fr is not None:
                frames.append(fr)

    def finish() -> dict:
        # Prefer a fresh evaluate() so red_toggled/blue_toggled are current.
        try:
            base = env.unwrapped
            while hasattr(base, "env"):
                # unwrap wrappers until we hit the MetaFine env
                nxt = getattr(base, "unwrapped", None)
                if nxt is base or nxt is None:
                    break
                base = nxt
            info = base.evaluate() if hasattr(base, "evaluate") else last_info
        except Exception:
            info = last_info
        toggled = _toggled_color(info)
        contact = info.get("had_contact", had_contact)
        if hasattr(contact, "item"):
            contact = bool(contact.item())
        return {
            "frames": frames,
            "toggled": toggled,
            "had_contact": bool(contact),
            "red_toggled": bool(
                info.get("red_toggled").item()
                if hasattr(info.get("red_toggled"), "item")
                else info.get("red_toggled", False)
            ),
            "blue_toggled": bool(
                info.get("blue_toggled").item()
                if hasattr(info.get("blue_toggled"), "item")
                else info.get("blue_toggled", False)
            ),
        }

    return on_step, finish


def _video_name(
    *,
    seed: int,
    success: bool,
    contact: bool,
    length: int,
    tag: Optional[str] = None,
) -> str:
    parts = [f"seed{int(seed):06d}"]
    if tag:
        parts.append(str(tag))
    parts.append(f"succ{int(bool(success))}")
    parts.append(f"contact{int(bool(contact))}")
    parts.append(f"len{int(length):03d}")
    return "_".join(parts) + ".mp4"


def run_perception(
    cfg: dict,
    policy,
    preprocessor,
    postprocessor,
    seeds: List[int],
    *,
    device: str,
    record_dir: Path,
    save_video: bool = False,
    video_fps: int = 30,
    perception_profiles: Optional[List[str]] = None,
) -> dict:
    perc = cfg.get("perception") or {}
    variant = perc.get("default_variant", "red")
    instruction = (cfg.get("variants") or {})[variant]
    max_steps = int(cfg.get("max_episode_steps", 300))

    base_kw = build_base_env_kwargs(cfg, extra=_variant_extra(cfg, variant))
    profiles = build_perception_profiles(base_kw, cfg)
    if perception_profiles:
        keep = set(perception_profiles)
        missing = keep - set(profiles)
        if missing:
            raise ValueError(
                f"--perception-profiles unknown: {sorted(missing)}; "
                f"available={sorted(profiles)}"
            )
        profiles = {k: v for k, v in profiles.items() if k in keep}

    n = min(int(perc.get("n_episodes", len(seeds))), len(seeds))
    use_seeds = seeds[:n]

    profile_results: Dict[str, Any] = {}
    for name, kw in profiles.items():
        successes = []
        episodes = []
        video_dir = record_dir / "videos" / f"perception_{name}"
        env = _make_env(kw)
        try:
            pbar = tqdm(use_seeds, desc=f"perception[{name}]")
            for seed in pbar:
                on_step, finish = _episode_probe(env, collect_frames=save_video)
                ep = run_episode(
                    env, policy, preprocessor, postprocessor,
                    task=instruction, seed=seed, max_steps=max_steps,
                    on_step=on_step,
                )
                diag = finish()
                frames = diag.pop("frames", [])
                detail = {
                    "seed": seed,
                    "success": ep["success"],
                    "episode_length": ep["episode_length"],
                    "toggled": diag["toggled"],
                    "had_contact": diag["had_contact"],
                    "red_toggled": diag["red_toggled"],
                    "blue_toggled": diag["blue_toggled"],
                }
                if save_video and frames:
                    vpath = video_dir / _video_name(
                        seed=seed,
                        success=ep["success"],
                        contact=diag["had_contact"],
                        length=ep["episode_length"],
                        tag=variant,
                    )
                    save_rgb_video(frames, vpath, fps=video_fps)
                    detail["video"] = str(vpath.relative_to(record_dir))
                successes.append(ep["success"])
                episodes.append(detail)
                pbar.set_postfix(sr=f"{np.mean(successes):.3f}")
        finally:
            try:
                env.close()
            except Exception:
                pass

        profile_results[name] = {
            "episodes": len(successes),
            "successes": int(np.sum(successes)),
            "success_rate": float(np.mean(successes)) if successes else float("nan"),
            "mean_episode_length": float(
                np.mean([e["episode_length"] for e in episodes])
            )
            if episodes
            else float("nan"),
            "details": episodes,
        }
        print(
            f"  {name}: sr={profile_results[name]['success_rate']:.3f} "
            f"mean_len={profile_results[name]['mean_episode_length']:.1f}"
        )

    summary = summarise_perception(profile_results)
    summary["meta"] = {
        "task_id": cfg.get("task_id"),
        "mode": "perception",
        "variant": variant,
        "instruction": instruction,
        "n_seeds": n,
        "seeds": use_seeds,
        "save_video": bool(save_video),
        "perception_profiles": list(profiles.keys()),
    }
    out = record_dir / "perception_summary.json"
    dump_json(out, summary)
    print(f"saved {out}")
    if "ausc_camera" in summary:
        print(f"AUSC(camera)={summary['ausc_camera']['value']:.4f}")
    if "ausc_light" in summary:
        print(f"AUSC(light)={summary['ausc_light']['value']:.4f}")
    if "ausc_mean" in summary:
        print(f"AUSC(mean)={summary['ausc_mean']['value']:.4f}")
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
) -> dict:
    variants: Dict[str, str] = dict(cfg.get("variants") or {})
    max_steps = int(cfg.get("max_episode_steps", 300))

    confusion: Dict[str, Dict[str, int]] = {
        v: defaultdict(int) for v in variants
    }
    per_variant_success: Dict[str, List[bool]] = {v: [] for v in variants}
    episodes: List[dict] = []

    envs = {
        variant: _make_env(build_base_env_kwargs(cfg, extra=_variant_extra(cfg, variant)))
        for variant in variants
    }
    try:
        for seed in tqdm(seeds, desc="understanding"):
            for variant, instruction in variants.items():
                on_step, finish = _episode_probe(
                    envs[variant], collect_frames=save_video,
                )
                ep = run_episode(
                    envs[variant], policy, preprocessor, postprocessor,
                    task=instruction, seed=seed, max_steps=max_steps,
                    on_step=on_step,
                )
                diag = finish()
                frames = diag.pop("frames", [])
                toggled = diag["toggled"]
                confusion[variant][toggled] += 1
                per_variant_success[variant].append(ep["success"])
                detail = {
                    "seed": seed,
                    "instructed": variant,
                    "instruction": instruction,
                    "success": ep["success"],
                    "toggled": toggled,
                    "had_contact": diag["had_contact"],
                    "red_toggled": diag["red_toggled"],
                    "blue_toggled": diag["blue_toggled"],
                    "episode_length": ep["episode_length"],
                }
                if save_video and frames:
                    vpath = (
                        record_dir / "videos" / "understanding" / variant
                        / _video_name(
                            seed=seed,
                            success=ep["success"],
                            contact=diag["had_contact"],
                            length=ep["episode_length"],
                        )
                    )
                    save_rgb_video(frames, vpath, fps=video_fps)
                    detail["video"] = str(vpath.relative_to(record_dir))
                episodes.append(detail)
    finally:
        for e in envs.values():
            try:
                e.close()
            except Exception:
                pass

    n_total = len(episodes)
    n_ok = sum(1 for e in episodes if e["success"])
    summary = {
        "meta": {
            "task_id": cfg.get("task_id"),
            "mode": "understanding",
            "n_seeds": len(seeds),
            "seeds": seeds,
            "variants": list(variants.keys()),
            "save_video": bool(save_video),
        },
        "success_rate": float(n_ok / n_total) if n_total else float("nan"),
        "n_episodes": n_total,
        "n_successes": n_ok,
        "mean_episode_length": float(
            np.mean([e["episode_length"] for e in episodes])
        )
        if episodes
        else float("nan"),
        "per_variant_success_rate": {
            v: float(np.mean(s)) if s else float("nan")
            for v, s in per_variant_success.items()
        },
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "episodes": episodes,
        "behavior": {
            "success_rate": float(n_ok / n_total) if n_total else float("nan"),
            "note": "single-stage task: Behavior = final success rate",
        },
    }
    out = record_dir / "understanding_summary.json"
    dump_json(out, summary)
    print(f"saved {out}")
    print(f"Understanding SR={summary['success_rate']:.3f}")
    print(f"per-variant: {summary['per_variant_success_rate']}")
    print(f"confusion: {summary['confusion']}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--policy-path", required=True)
    p.add_argument("--seeds", required=True, help="JSON from select_eval_seeds.py")
    p.add_argument(
        "--mode", choices=["perception", "understanding", "both"], default="both"
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--record-dir", default="eval_runs/toggle_switch_table")
    p.add_argument("--n-seeds", type=int, default=None, help="Use only the first N seeds")
    p.add_argument(
        "--save-video",
        action="store_true",
        help="Save side-by-side base+hand camera mp4 per episode",
    )
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument(
        "--perception-profiles",
        default=None,
        help="Comma-separated Perception profile names (default: all).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_task_config(args.config)
    seeds = load_seed_list(args.seeds)
    if args.n_seeds is not None:
        seeds = seeds[: args.n_seeds]
    record_dir = resolve_path(args.record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)

    perc_profiles = None
    if args.perception_profiles:
        perc_profiles = [
            s.strip() for s in args.perception_profiles.split(",") if s.strip()
        ]

    print(f"config={args.config}")
    print(f"policy={args.policy_path}")
    print(f"seeds={len(seeds)} from {args.seeds}")
    print(f"sensor={cfg.get('sensor')}")
    print(f"save_video={args.save_video} video_fps={args.video_fps}")
    print(f"perception_profiles={perc_profiles or 'ALL'}")

    policy, preprocessor, postprocessor = load_pi0_policy(
        args.policy_path, device=args.device, tokenizer_path=args.tokenizer_path,
    )

    results = {
        "task_id": cfg.get("task_id"),
        "seeds_file": str(resolve_path(args.seeds)),
    }
    if args.mode in ("perception", "both"):
        results["perception"] = run_perception(
            cfg,
            policy,
            preprocessor,
            postprocessor,
            seeds,
            device=args.device,
            record_dir=record_dir,
            save_video=args.save_video,
            video_fps=args.video_fps,
            perception_profiles=perc_profiles,
        )
    if args.mode in ("understanding", "both"):
        results["understanding"] = run_understanding(
            cfg,
            policy,
            preprocessor,
            postprocessor,
            seeds,
            record_dir=record_dir,
            save_video=args.save_video,
            video_fps=args.video_fps,
        )
    dump_json(
        record_dir / "run_summary.json",
        {
            k: (v.get("meta") if isinstance(v, dict) and "meta" in v else v)
            for k, v in results.items()
        },
    )


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
