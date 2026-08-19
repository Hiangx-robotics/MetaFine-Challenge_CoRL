#!/usr/bin/env python3
"""Evaluate a PI0 policy on ``grasp_part`` (T1).

Two modes (can be combined in one run):

* **Perception** (``--mode perception``): sweep camera (pos+rot joint) and
  light DR levels; report per-level success rates + AUSC.
* **Understanding** (``--mode understanding``): fix each seed, run every
  instruction variant (cap / body); report success rate + confusion matrix.
  Success requires the correct part (``eval_require_correct_part=True``).

Sensor settings are read from ``eval/configs/grasp_part.yaml`` so they stay
aligned with the training replay metadata (512×512, fov≈70°, shader=default).
Do **not** rely on env defaults (224×224).

Optional diagnostics (``--save-video``):
  side-by-side policy-view mp4 + contact verification via ``agent.is_grasping``,
  so false-positive "air grasps" (gripper half-closed with no contact) can be
  quantified without changing the existing success definition.

Seeds must be pre-selected and saved (RoboTwin-style)::

  python -m eval.select_eval_seeds --config eval/configs/grasp_part.yaml
  python -m eval.eval_grasp_part \\
      --policy-path outputs/pi0_grasp_mixed_f70/checkpoints/030000/pretrained_model \\
      --seeds /tmp/grasp_part_dev.json \\
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
    obs_to_frame,
    resolve_path,
    run_episode,
    save_rgb_video,
    summarise_perception,
)

DEFAULT_CONFIG = "eval/configs/grasp_part.yaml"


def _make_env(env_kwargs: dict):
    env = gym.make("grasp_part", **env_kwargs)
    return CPUGymWrapper(env, ignore_terminations=True, record_metrics=True)


def _obj_root_pos(base) -> Optional[np.ndarray]:
    obj = getattr(base, "target_object", None)
    if obj is None:
        return None
    try:
        pose = obj.pose if hasattr(obj, "pose") else obj.get_pose()
        p = np.asarray(pose.p).reshape(-1)[:3].astype(np.float64)
        return p
    except Exception:
        return None


def _contact_link_name(base) -> Optional[str]:
    """Return first link name for which ``agent.is_grasping(link)`` is True."""
    obj = getattr(base, "target_object", None)
    agent = getattr(base, "agent", None)
    if obj is None or agent is None or not hasattr(agent, "is_grasping"):
        return None
    try:
        links = obj.get_links()
    except Exception:
        return None
    for link in links:
        try:
            grasped = agent.is_grasping(link)
            if isinstance(grasped, (list, tuple)):
                grasped = grasped[0]
            if hasattr(grasped, "item"):
                grasped = bool(grasped.item())
            else:
                grasped = bool(grasped)
            if grasped:
                name = getattr(link, "name", None) or getattr(link, "key", None)
                return str(name) if name is not None else "link"
        except Exception:
            continue
    return None


def _episode_probe(
    env,
    *,
    collect_frames: bool = False,
) -> Tuple[Callable[[int, dict, Any], None], Callable[[], dict]]:
    """Factory: ``(on_step, finish)`` for contact / TCP / displacement / frames.

    ``init_pos`` is captured on the first ``on_step`` (after the episode's
    ``env.reset`` inside ``run_episode``), not at factory time — the env has
    not been reset yet when this factory runs.
    """
    base = env.unwrapped
    init_pos: Optional[np.ndarray] = None
    frames: List[np.ndarray] = []
    last: Dict[str, Any] = {
        "contact_link": None,
        "tcp_part": None,
        "tcp_part_dist": float("inf"),
        "step": -1,
    }
    contact_ever = False
    contact_link_ever: Optional[str] = None

    def on_step(t: int, obs: dict, info: Any) -> None:
        nonlocal contact_ever, contact_link_ever, init_pos
        # First post-reset step ≈ initial object pose for displacement.
        if init_pos is None:
            init_pos = _obj_root_pos(base)
        link = _contact_link_name(base)
        if link is not None:
            contact_ever = True
            contact_link_ever = link
        try:
            part, dist = base._nearest_annotated_part()
        except Exception:
            part, dist = None, float("inf")
        last.update({
            "contact_link": link,
            "tcp_part": part,
            "tcp_part_dist": float(dist) if dist is not None else float("inf"),
            "step": int(t),
        })
        if collect_frames:
            try:
                frames.append(obs_to_frame(obs))
            except Exception:
                pass

    def finish() -> dict:
        end_pos = _obj_root_pos(base)
        if init_pos is not None and end_pos is not None:
            obj_disp = float(np.linalg.norm(end_pos - init_pos))
        else:
            obj_disp = float("nan")
        contact_at_success = last["contact_link"] is not None
        return {
            "contact_at_success": bool(contact_at_success),
            "contact_ever": bool(contact_ever),
            "contact_link": last["contact_link"] or contact_link_ever,
            "tcp_part_at_success": last["tcp_part"],
            "tcp_part_dist_at_success": (
                None if not np.isfinite(last["tcp_part_dist"])
                else float(last["tcp_part_dist"])
            ),
            "obj_disp": obj_disp,
            "frames": frames,
        }

    return on_step, finish


def _diag_aggregate(episodes: List[dict]) -> dict:
    """Aggregate contact / length diagnostics across episode detail dicts."""
    n = len(episodes)
    if n == 0:
        return {
            "contact_success_rate": float("nan"),
            "false_positive_rate": float("nan"),
            "mean_episode_length": float("nan"),
            "mean_tcp_part_dist_at_success": float("nan"),
            "mean_obj_disp": float("nan"),
            "n_contact_success": 0,
            "n_false_positive": 0,
        }
    n_succ = sum(1 for e in episodes if e.get("success"))
    n_contact_succ = sum(
        1 for e in episodes if e.get("success") and e.get("contact_at_success")
    )
    n_fp = sum(
        1 for e in episodes if e.get("success") and not e.get("contact_at_success")
    )
    lengths = [e.get("episode_length", 0) for e in episodes]
    dists = [
        e["tcp_part_dist_at_success"]
        for e in episodes
        if e.get("success") and e.get("tcp_part_dist_at_success") is not None
    ]
    disps = [
        e["obj_disp"]
        for e in episodes
        if e.get("obj_disp") is not None and np.isfinite(e["obj_disp"])
    ]
    sr = n_succ / n
    csr = n_contact_succ / n
    fpr = (1.0 - csr / sr) if sr > 0 else float("nan")
    return {
        "contact_success_rate": float(csr),
        "false_positive_rate": float(fpr) if np.isfinite(fpr) else float("nan"),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_tcp_part_dist_at_success": float(np.mean(dists)) if dists else float("nan"),
        "mean_obj_disp": float(np.mean(disps)) if disps else float("nan"),
        "n_contact_success": int(n_contact_succ),
        "n_false_positive": int(n_fp),
    }


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
    variant = perc.get("default_variant", "cap")
    instruction = (cfg.get("variants") or {})[variant]
    object_name = cfg.get("object_name", "3558")
    max_steps = int(cfg.get("max_episode_steps", 300))

    base_kw = build_base_env_kwargs(
        cfg,
        extra={
            "object_name": object_name,
            "part_name": variant,
            # Same strict success criterion as Understanding (contact + part).
            "eval_require_correct_part": True,
        },
    )
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

    # Cap episodes to available seeds.
    n = min(int(perc.get("n_episodes", len(seeds))), len(seeds))
    use_seeds = seeds[:n]

    profile_results: Dict[str, Any] = {}
    for name, kw in profiles.items():
        successes = []
        episodes = []
        video_dir = record_dir / "videos" / f"perception_{name}"
        # Reuse one env per profile — repeated gym.make/close segfaults SAPIEN.
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
                    "grasped_part": ep["extras"].get("grasped_part"),
                    "contact_at_success": diag["contact_at_success"],
                    "contact_ever": diag["contact_ever"],
                    "contact_link": diag["contact_link"],
                    "tcp_part_at_success": diag["tcp_part_at_success"],
                    "tcp_part_dist_at_success": diag["tcp_part_dist_at_success"],
                    "obj_disp": diag["obj_disp"],
                }
                if save_video and frames:
                    vpath = video_dir / _video_name(
                        seed=seed,
                        success=ep["success"],
                        contact=diag["contact_at_success"],
                        length=ep["episode_length"],
                        tag=variant,
                    )
                    save_rgb_video(frames, vpath, fps=video_fps)
                    detail["video"] = str(vpath.relative_to(record_dir))
                successes.append(ep["success"])
                episodes.append(detail)
                pbar.set_postfix(
                    sr=f"{np.mean(successes):.3f}",
                    fp=sum(1 for e in episodes if e["success"] and not e["contact_at_success"]),
                )
        finally:
            env.close()

        agg = _diag_aggregate(episodes)
        profile_results[name] = {
            "episodes": len(successes),
            "successes": int(np.sum(successes)),
            "success_rate": float(np.mean(successes)) if successes else float("nan"),
            **agg,
            "details": episodes,
        }
        print(
            f"  {name}: sr={profile_results[name]['success_rate']:.3f} "
            f"contact_sr={agg['contact_success_rate']:.3f} "
            f"fp_rate={agg['false_positive_rate']:.3f} "
            f"mean_len={agg['mean_episode_length']:.1f}"
        )

    # summarise_perception keeps per-profile dicts as-is (incl. diagnostics).
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
    object_name = cfg.get("object_name", "3558")
    max_steps = int(cfg.get("max_episode_steps", 300))

    # confusion[instructed][grasped_or_none] = count
    confusion: Dict[str, Dict[str, int]] = {
        v: defaultdict(int) for v in variants
    }
    per_variant_success: Dict[str, List[bool]] = {v: [] for v in variants}
    per_variant_contact: Dict[str, List[bool]] = {v: [] for v in variants}
    episodes: List[dict] = []

    # One env per variant (part_name is a constructor kwarg).
    envs = {
        variant: _make_env(build_base_env_kwargs(
            cfg,
            extra={
                "object_name": object_name,
                "part_name": variant,
                "eval_require_correct_part": True,
            },
        ))
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
                grasped = ep["extras"].get("grasped_part") or "none"
                confusion[variant][grasped] += 1
                per_variant_success[variant].append(ep["success"])
                per_variant_contact[variant].append(
                    bool(ep["success"] and diag["contact_at_success"])
                )
                detail = {
                    "seed": seed,
                    "instructed": variant,
                    "instruction": instruction,
                    "success": ep["success"],
                    "grasped_part": grasped,
                    "episode_length": ep["episode_length"],
                    "contact_at_success": diag["contact_at_success"],
                    "contact_ever": diag["contact_ever"],
                    "contact_link": diag["contact_link"],
                    "tcp_part_at_success": diag["tcp_part_at_success"],
                    "tcp_part_dist_at_success": diag["tcp_part_dist_at_success"],
                    "obj_disp": diag["obj_disp"],
                }
                if save_video and frames:
                    vpath = (
                        record_dir / "videos" / "understanding" / variant
                        / _video_name(
                            seed=seed,
                            success=ep["success"],
                            contact=diag["contact_at_success"],
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
    agg = _diag_aggregate(episodes)
    summary = {
        "meta": {
            "task_id": cfg.get("task_id"),
            "mode": "understanding",
            "n_seeds": len(seeds),
            "seeds": seeds,
            "variants": list(variants.keys()),
            "eval_require_correct_part": True,
            "save_video": bool(save_video),
        },
        "success_rate": float(n_ok / n_total) if n_total else float("nan"),
        "n_episodes": n_total,
        "n_successes": n_ok,
        **agg,
        "per_variant_success_rate": {
            v: float(np.mean(s)) if s else float("nan")
            for v, s in per_variant_success.items()
        },
        "per_variant_contact_success_rate": {
            v: float(np.mean(s)) if s else float("nan")
            for v, s in per_variant_contact.items()
        },
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "episodes": episodes,
        # Behavior for single-stage grasp_part ≡ final task success.
        "behavior": {
            "success_rate": float(n_ok / n_total) if n_total else float("nan"),
            "contact_success_rate": agg["contact_success_rate"],
            "false_positive_rate": agg["false_positive_rate"],
            "note": "single-stage task: Behavior = final success rate",
        },
    }
    out = record_dir / "understanding_summary.json"
    dump_json(out, summary)
    print(f"saved {out}")
    print(f"Understanding SR={summary['success_rate']:.3f}")
    print(f"contact_sr={agg['contact_success_rate']:.3f} fp_rate={agg['false_positive_rate']:.3f}")
    print(f"per-variant: {summary['per_variant_success_rate']}")
    print(f"per-variant contact: {summary['per_variant_contact_success_rate']}")
    print(f"confusion: {summary['confusion']}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--policy-path", required=True)
    p.add_argument("--seeds", required=True, help="JSON from select_eval_seeds.py")
    p.add_argument("--mode", choices=["perception", "understanding", "both"], default="both")
    p.add_argument("--device", default="cuda")
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--record-dir", default="eval_runs/grasp_part")
    p.add_argument("--n-seeds", type=int, default=None, help="Use only the first N seeds")
    p.add_argument(
        "--save-video", action="store_true",
        help="Save side-by-side base+hand camera mp4 per episode + contact diagnostics",
    )
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument(
        "--perception-profiles", default=None,
        help="Comma-separated Perception profile names to run (e.g. 'clean'). "
             "Default: all profiles from the task YAML.",
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
        perc_profiles = [s.strip() for s in args.perception_profiles.split(",") if s.strip()]

    print(f"config={args.config}")
    print(f"policy={args.policy_path}")
    print(f"seeds={len(seeds)} from {args.seeds}")
    print(f"sensor={cfg.get('sensor')}")
    print(f"save_video={args.save_video} video_fps={args.video_fps}")
    print(f"perception_profiles={perc_profiles or 'ALL'}")

    policy, preprocessor, postprocessor = load_pi0_policy(
        args.policy_path, device=args.device, tokenizer_path=args.tokenizer_path,
    )

    results = {"task_id": cfg.get("task_id"), "seeds_file": str(resolve_path(args.seeds))}
    if args.mode in ("perception", "both"):
        results["perception"] = run_perception(
            cfg, policy, preprocessor, postprocessor, seeds,
            device=args.device, record_dir=record_dir,
            save_video=args.save_video, video_fps=args.video_fps,
            perception_profiles=perc_profiles,
        )
    if args.mode in ("understanding", "both"):
        results["understanding"] = run_understanding(
            cfg, policy, preprocessor, postprocessor, seeds,
            record_dir=record_dir,
            save_video=args.save_video, video_fps=args.video_fps,
        )
    dump_json(record_dir / "run_summary.json", {
        k: (v.get("meta") if isinstance(v, dict) and "meta" in v else v)
        for k, v in results.items()
    })


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
