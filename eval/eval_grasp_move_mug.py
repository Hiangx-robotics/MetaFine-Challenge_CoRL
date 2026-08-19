#!/usr/bin/env python3
"""Evaluate a PI0 policy on ``grasp_move_mug`` (T2).

* **Perception**: camera (pos+rot joint) + light DR sweeps → AUSC, using
  the default instruction (left).
* **Understanding**: fixed seeds × {left, right, forward} instructions;
  env is rebuilt from the matching YAML task graph so ``evaluate()``
  scores the instructed world-frame translation.

Success is the training-time predicate: grasped handle + TCP travelled
≥7 cm and mug ≥5 cm along the instructed axis. ``mark_move_intent`` is
installed on ``env.reset`` because the policy never calls the planner
skill that would otherwise publish the axis.

Sensor settings come from ``eval/configs/grasp_move_mug.yaml``
(512×512, fov≈70°, shader=default) — matching the training replay.

Seeds must be pre-selected::

  python -m eval.select_eval_seeds --config eval/configs/grasp_move_mug.yaml
  python -m eval.eval_grasp_move_mug \\
      --policy-path outputs/pi0_grasp_move_mug_mixed/checkpoints/030000/pretrained_model \\
      --seeds /tmp/grasp_move_mug_dev.json \\
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

import numpy as np
from tqdm import tqdm
from mani_skill.utils.wrappers import CPUGymWrapper

from utils.eval_common import (
    build_perception_profiles,
    dump_json,
    load_pi0_policy,
    load_seed_list,
    load_task_config,
    obs_to_frame,
    resolve_path,
    run_episode,
    save_rgb_video,
    sensor_configs_from_task,
    summarise_perception,
)
from utils.eval_setup import make_eval_env

DEFAULT_CONFIG = "eval/configs/grasp_move_mug.yaml"

# Image-frame instruction → world-frame move_to_direction axis.
# Matches configs/t2_mug_move_{left,right,forward}.yaml.
VARIANT_MOVE = {
    "left": {"direction": "backward", "axis": (0.0, -1.0, 0.0), "step": 0.10},
    "right": {"direction": "forward", "axis": (0.0, 1.0, 0.0), "step": 0.10},
    "forward": {"direction": "right", "axis": (1.0, 0.0, 0.0), "step": 0.10},
}


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


def _as_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return float(v[0]) if v else None
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except Exception:
            pass
    try:
        return float(v)
    except Exception:
        return None


def _install_move_intent(env, variant: str) -> None:
    """After reset, publish the instructed translation axis for evaluate()."""
    spec = VARIANT_MOVE[variant]
    orig = env.reset

    def reset(*args, **kwargs):
        obs, info = orig(*args, **kwargs)
        env.unwrapped.mark_move_intent(
            axis=np.asarray(spec["axis"], dtype=np.float64),
            step=float(spec["step"]),
            direction=str(spec["direction"]),
        )
        return obs, info

    env.reset = reset


def _make_env(cfg: dict, variant: str, extra: Optional[dict] = None):
    graphs = cfg.get("task_graphs") or {}
    if variant not in graphs:
        raise KeyError(f"no task_graph for variant {variant!r}")
    args = {
        "task_graph": graphs[variant],
        "obs_mode": cfg.get("obs_mode", "rgb"),
        "control_mode": cfg.get("control_mode", "pd_joint_delta_pos"),
        "render_mode": "rgb_array",
        "sim_backend": cfg.get("sim_backend", "physx_cpu"),
        "sensor_configs": sensor_configs_from_task(cfg),
        "human_render_camera_configs": dict(
            shader_pack=(cfg.get("sensor") or {}).get("shader_pack", "default")
        ),
        "viewer_camera_configs": dict(
            shader_pack=(cfg.get("sensor") or {}).get("shader_pack", "default")
        ),
    }
    extra_kw: dict[str, Any] = {
        "eval_require_correct_part": False,
        "object_name": cfg.get("object_name", "8848"),
        "part_name": cfg.get("part_name", "handle"),
        # A policy never calls engage(), so score the grasp from contact.
        "grasped_contact_fallback": True,
    }
    if cfg.get("max_episode_steps") is not None:
        extra_kw["max_episode_steps"] = int(cfg["max_episode_steps"])
    if extra:
        extra_kw.update(extra)
    env, _ = make_eval_env(args, extra_env_kwargs=extra_kw)
    wrapped = CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)
    _install_move_intent(wrapped, variant)
    return wrapped


def _obj_pos(env) -> Optional[np.ndarray]:
    obj = getattr(env.unwrapped, "target_object", None)
    if obj is None:
        return None
    try:
        p = np.asarray(obj.pose.p).reshape(-1)[:3].astype(np.float64)
        return p
    except Exception:
        return None


def _classify_xy(delta: Optional[np.ndarray], *, min_dist: float = 0.03) -> str:
    """Map world-frame mug XY displacement onto {left, right, forward, none}."""
    if delta is None:
        return "none"
    scores = {
        "left": float(-delta[1]),
        "right": float(delta[1]),
        "forward": float(delta[0]),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] >= min_dist else "none"


def _episode_probe(env, variant: str, *, part_name: str, collect_frames: bool = False):
    """Track mug pose / grasp onset, and optionally collect policy-view frames.

    The translation origin is re-armed the first time the gripper actually
    holds the handle, so ``moved_along`` measures the carry leg only — the
    planner-recorded demos are scored the same way (``mark_move_intent`` runs
    after ``grasp_part``), whereas arming at reset would also count reaching.
    """
    spec = VARIANT_MOVE[variant]
    frames: List[np.ndarray] = []
    start: Optional[np.ndarray] = None
    end: Optional[np.ndarray] = None
    grasp_step: Optional[int] = None
    grasped_ever = False

    def on_step(t: int, obs: Any, _info: Any) -> None:
        nonlocal start, end, grasp_step, grasped_ever
        base = env.unwrapped

        p = _obj_pos(env)
        if p is not None:
            if start is None:
                start = p.copy()
            end = p.copy()

        try:
            holding = bool(base.grasped(part_name))
        except Exception:
            holding = False
        if holding:
            grasped_ever = True
            if grasp_step is None:
                grasp_step = int(t)
                base.mark_move_intent(
                    axis=np.asarray(spec["axis"], dtype=np.float64),
                    step=float(spec["step"]),
                    direction=str(spec["direction"]),
                )

        if collect_frames:
            try:
                frames.append(obs_to_frame(obs))
            except Exception:
                pass

    def finish() -> dict:
        delta = None if start is None or end is None else (end - start)
        return {
            "frames": frames,
            "obj_start": None if start is None else start.tolist(),
            "obj_end": None if end is None else end.tolist(),
            "obj_delta": None if delta is None else delta.tolist(),
            "moved": _classify_xy(delta),
            "grasp_step": grasp_step,
            "grasped_ever": grasped_ever,
        }

    return on_step, finish


def _video_name(*, seed: int, variant: str, success: bool, grasped: bool, length: int) -> str:
    return (
        f"seed{int(seed):06d}_{variant}_succ{int(bool(success))}"
        f"_grasp{int(bool(grasped))}_len{int(length):03d}.mp4"
    )


def _stage_id(extras: dict) -> int:
    ev = extras.get("evaluate") or {}
    n = 0
    if _as_bool(ev.get("stage_grasped")):
        n = 1
    if _as_bool(ev.get("stage_moved")):
        n = 2
    return n


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
    variant = perc.get("default_variant", "left")
    instruction = (cfg.get("variants") or {})[variant]
    max_steps = int(cfg.get("max_episode_steps", 400))
    part_name = cfg.get("part_name", "handle")

    base_kw = {
        "eval_require_correct_part": False,
        "object_name": cfg.get("object_name", "8848"),
        "part_name": cfg.get("part_name", "handle"),
    }
    if cfg.get("max_episode_steps") is not None:
        base_kw["max_episode_steps"] = int(cfg["max_episode_steps"])
    profiles = build_perception_profiles(base_kw, cfg)
    n = min(int(perc.get("n_episodes", len(seeds))), len(seeds))
    use_seeds = seeds[:n]

    profile_results: Dict[str, Any] = {}
    for name, extra in profiles.items():
        successes, stage_ids, episodes = [], [], []
        # Videos only for the clean profile — the DR sweeps would multiply
        # disk for little extra insight.
        want_video = bool(save_video) and name == "clean"
        video_dir = record_dir / "videos" / f"perception_{name}"
        n_saved = 0
        env = _make_env(cfg, variant, extra=extra)
        try:
            pbar = tqdm(use_seeds, desc=f"perception[{name}]")
            for seed in pbar:
                grab = want_video and n_saved < int(max_videos)
                on_step, finish = _episode_probe(
                    env, variant, part_name=part_name, collect_frames=grab,
                )
                ep = run_episode(
                    env, policy, preprocessor, postprocessor,
                    task=instruction, seed=seed, max_steps=max_steps,
                    on_step=on_step,
                )
                diag = finish()
                frames = diag.pop("frames", [])
                sid = _stage_id(ep["extras"])
                successes.append(ep["success"])
                stage_ids.append(sid)
                detail = {
                    "seed": seed,
                    "success": ep["success"],
                    "episode_length": ep["episode_length"],
                    "stage_id": sid,
                    "moved": diag["moved"],
                    "obj_delta": diag["obj_delta"],
                    "grasp_step": diag["grasp_step"],
                    "grasped_ever": diag["grasped_ever"],
                    "moved_object": _as_float(
                        (ep["extras"].get("evaluate") or {}).get("moved_object")
                    ),
                    "moved_tcp": _as_float(
                        (ep["extras"].get("evaluate") or {}).get("moved_tcp")
                    ),
                }
                if grab and frames:
                    vpath = video_dir / _video_name(
                        seed=seed, variant=variant, success=ep["success"],
                        grasped=diag["grasped_ever"], length=ep["episode_length"],
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
    save_video: bool = False,
    video_fps: int = 30,
    max_videos: int = 6,
) -> dict:
    variants: Dict[str, str] = dict(cfg.get("variants") or {})
    max_steps = int(cfg.get("max_episode_steps", 400))
    part_name = cfg.get("part_name", "handle")
    n_saved: Dict[str, int] = {v: 0 for v in variants}

    confusion: Dict[str, Dict[str, int]] = {v: defaultdict(int) for v in variants}
    per_variant_success: Dict[str, List[bool]] = {v: [] for v in variants}
    stage_hist: Dict[int, int] = defaultdict(int)
    episodes: List[dict] = []

    envs = {variant: _make_env(cfg, variant) for variant in variants}
    try:
        for seed in tqdm(seeds, desc="understanding"):
            for variant, instruction in variants.items():
                grab = bool(save_video) and n_saved[variant] < int(max_videos)
                on_step, finish = _episode_probe(
                    envs[variant], variant, part_name=part_name, collect_frames=grab,
                )
                ep = run_episode(
                    envs[variant], policy, preprocessor, postprocessor,
                    task=instruction, seed=seed, max_steps=max_steps,
                    on_step=on_step,
                )
                diag = finish()
                frames = diag.pop("frames", [])
                moved = diag["moved"]
                confusion[variant][moved] += 1
                per_variant_success[variant].append(ep["success"])
                sid = _stage_id(ep["extras"])
                stage_hist[sid] += 1
                detail = {
                    "seed": seed,
                    "instructed": variant,
                    "instruction": instruction,
                    "success": ep["success"],
                    "moved": moved,
                    "stage_id": sid,
                    "episode_length": ep["episode_length"],
                    "obj_delta": diag["obj_delta"],
                    "grasp_step": diag["grasp_step"],
                    "grasped_ever": diag["grasped_ever"],
                    "moved_object": _as_float(
                        (ep["extras"].get("evaluate") or {}).get("moved_object")
                    ),
                    "moved_tcp": _as_float(
                        (ep["extras"].get("evaluate") or {}).get("moved_tcp")
                    ),
                }
                if grab and frames:
                    vpath = record_dir / "videos" / "understanding" / variant / _video_name(
                        seed=seed, variant=variant, success=ep["success"],
                        grasped=diag["grasped_ever"], length=ep["episode_length"],
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
    stage_fracs = [e["stage_id"] / 2.0 for e in episodes]
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
            "note": "two-stage: Behavior reports final SR + mean(stage_id/2)",
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
    p.add_argument("--record-dir", default="eval_runs/grasp_move_mug")
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
