#!/usr/bin/env python3
"""Select planner-solvable eval seeds that are disjoint from the training set.

RoboTwin-style protocol:
  1. Load ``episode_seed`` values from the task's training demo JSONs.
  2. Sample candidate seeds outside that set.
  3. Run the motion-planning expert (``obs_mode=none``, no RGB) for every
     instruction variant listed in the task YAML.
  4. Keep a seed only if every required variant plans to success.
  5. Write the list to a path you choose (e.g. ``/tmp/<task>_dev.json``).

**Competition release:** this repository ships **no** pre-generated seed files.
Official evaluation uses a hidden seed list from a private organizer ``--rng-seed``.
Local seeds you generate will differ from the competition set.

Examples::

  python -m eval.select_eval_seeds --config eval/configs/grasp_part.yaml
  python -m eval.select_eval_seeds --config eval/configs/put_blocks_into_boxes.yaml \\
      --n-seeds 20 --rng-seed 42 --out /tmp/put_blocks_dev.json
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path
from random import Random
from typing import Any, Optional

# Repo root on sys.path so ``import core / utils / eval`` works when launched
# as ``python eval/select_eval_seeds.py`` or ``python -m eval.select_eval_seeds``.
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

import core.env  # noqa: F401
import core.skill  # noqa: F401
from core.common import MP_SOLUTIONS
from mani_skill.examples.motionplanning.panda.motionplanner import (
    PandaArmMotionPlanningSolver,
)
from utils.eval_common import (
    load_task_config,
    load_train_seeds,
    resolve_path,
    save_seed_list,
)


def _variant_env_kwargs(cfg: dict, variant: str) -> dict:
    """Map a variant name → env constructor kwargs for this task."""
    env_id = cfg["env_id"]
    kw: dict[str, Any] = {}
    # Prefer an explicit YAML key (e.g. target_switch for toggle_switch_table).
    variant_key = cfg.get("variant_env_kwarg")
    if variant_key:
        if cfg.get("object_name") is not None:
            kw["object_name"] = cfg["object_name"]
        if cfg.get("part_name") is not None:
            kw["part_name"] = cfg["part_name"]
        kw[str(variant_key)] = variant
        return kw
    if env_id == "grasp_part":
        kw["object_name"] = cfg.get("object_name", "3558")
        kw["part_name"] = variant
    elif env_id == "put_blocks_into_boxes":
        kw["special_cube"] = variant
    elif env_id == "toggle_switch_table":
        kw["object_name"] = cfg.get("object_name", "100920")
        kw["part_name"] = cfg.get("part_name", "button")
        kw["target_switch"] = variant
    elif env_id == "insert_letter":
        kw["target_letter"] = variant
    else:
        # Generic fallback: treat variant as part_name when object_name is set.
        if cfg.get("object_name") is not None:
            kw["object_name"] = cfg["object_name"]
            kw["part_name"] = variant
    return kw


def _make_plan_env(cfg: dict, variant: str):
    """Build one lightweight (obs_mode=none) env for planner checks."""
    graphs = cfg.get("task_graphs") or {}
    if variant in graphs:
        from utils.eval_setup import make_eval_env

        args = {
            "task_graph": graphs[variant],
            "obs_mode": "none",
            "control_mode": "pd_joint_pos",
            "render_mode": "rgb_array",
            "sim_backend": cfg.get("sim_backend", "physx_cpu"),
            "sensor_configs": dict(shader_pack="default", width=224, height=224),
            "human_render_camera_configs": dict(shader_pack="default"),
            "viewer_camera_configs": dict(shader_pack="default"),
        }
        extra: dict[str, Any] = {"eval_require_correct_part": False}
        if cfg.get("object_name") is not None:
            extra["object_name"] = cfg["object_name"]
        if cfg.get("part_name") is not None:
            extra["part_name"] = cfg["part_name"]
        env, _ = make_eval_env(args, extra_env_kwargs=extra)
        return env
    return gym.make(
        cfg["env_id"],
        obs_mode="none",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack="default", width=224, height=224),
        human_render_camera_configs=dict(shader_pack="default"),
        viewer_camera_configs=dict(shader_pack="default"),
        sim_backend=cfg.get("sim_backend", "physx_cpu"),
        **_variant_env_kwargs(cfg, variant),
    )


def _plan_on_env(
    env,
    solve,
    cfg: dict,
    variant: str,
    seed: int,
    *,
    verbose: bool = False,
) -> bool:
    """Reset ``env`` to ``seed`` and run the motion planner. Env is reused."""
    import inspect

    planner = None
    try:
        env.reset(seed=int(seed))
        planner = PandaArmMotionPlanningSolver(
            env,
            vis=False,
            base_pose=env.unwrapped.agent.robot.pose,
            visualize_target_grasp_pose=False,
            print_env_info=False,
        )
        graphs = cfg.get("task_graphs") or {}
        if variant in graphs:
            from utils.task_graph import load_task_graph, run_task_graph

            tg = load_task_graph(graphs[variant])
            result = run_task_graph(planner, tg.to_chain(), verbose=verbose)
        else:
            sig = inspect.signature(solve)
            call_kwargs: dict[str, Any] = {}
            if "verbose" in sig.parameters:
                call_kwargs["verbose"] = verbose
            if "part_name" in sig.parameters:
                if cfg["env_id"] == "grasp_part":
                    call_kwargs["part_name"] = variant
                elif cfg.get("part_name") is not None:
                    call_kwargs["part_name"] = cfg["part_name"]
                elif cfg["env_id"] == "toggle_switch_table":
                    call_kwargs["part_name"] = "button"
            if "seed" in sig.parameters:
                call_kwargs["seed"] = seed
            result = solve(planner, **call_kwargs)

        if result == -1 or result is False:
            return False
        # Always gate on env.evaluate() when available — motion-plan success
        # alone can leave the object tipped / ungripped under the strict
        # grasp_part criterion.
        try:
            info = env.unwrapped.evaluate()
            s = info.get("success")
            if isinstance(s, torch.Tensor):
                return bool(s.detach().cpu().numpy().reshape(-1)[0])
            if s is not None:
                return bool(s)
        except Exception:
            pass
        if result is True:
            return True
        if isinstance(result, tuple) and len(result) >= 5 and isinstance(result[4], dict):
            s = result[4].get("success", False)
            if isinstance(s, torch.Tensor):
                return bool(s.detach().cpu().numpy().reshape(-1)[0])
            return bool(s)
        return result != -1
    except Exception:
        if verbose:
            traceback.print_exc()
        return False
    finally:
        if planner is not None:
            try:
                planner.close()
            except Exception:
                pass


def select_seeds(
    cfg: dict,
    *,
    n_seeds: int,
    seed_range: tuple[int, int],
    rng_seed: int = 0,
    verbose: bool = False,
    max_trials: Optional[int] = None,
) -> list[int]:
    """Sample until ``n_seeds`` planner-validated seeds are collected.

    Reuses one env per variant across candidates — repeatedly ``gym.make`` /
    ``close`` segfaults SAPIEN after ~10–20 cycles on this stack.
    """
    train_seeds = load_train_seeds(cfg.get("train_demo_jsons") or [])
    sel = cfg.get("seed_select") or {}
    variants = list(sel.get("check_variants") or list((cfg.get("variants") or {}).keys()))
    require_all = bool(sel.get("require_all_variants", True))
    if not variants:
        raise ValueError("no variants to check; set seed_select.check_variants in the YAML")

    env_id = cfg["env_id"]
    skill_type = sel.get("skill_type") or env_id
    graphs = cfg.get("task_graphs") or {}
    if graphs:
        solve = None
    else:
        solve = MP_SOLUTIONS.get(skill_type)
        if solve is None:
            raise KeyError(f"no MP solution registered for skill_type={skill_type!r}")

    # One env per variant, kept alive for the whole selection run.
    envs = {v: _make_plan_env(cfg, v) for v in variants}

    lo, hi = int(seed_range[0]), int(seed_range[1])
    rng = Random(rng_seed)
    accepted: list[int] = []
    rejected_train = 0
    rejected_plan = 0
    trials = 0
    max_trials = max_trials or max(n_seeds * 50, 500)

    try:
        pbar = tqdm(total=n_seeds, desc=f"select seeds [{cfg.get('task_id')}]")
        while len(accepted) < n_seeds and trials < max_trials:
            trials += 1
            cand = rng.randint(lo, hi - 1)
            if cand in train_seeds or cand in accepted:
                rejected_train += 1
                continue

            ok_flags = []
            for v in variants:
                ok = _plan_on_env(envs[v], solve, cfg, v, cand, verbose=verbose)
                ok_flags.append(ok)
                if require_all and not ok:
                    break

            if require_all:
                keep = all(ok_flags) and len(ok_flags) == len(variants)
            else:
                keep = any(ok_flags)

            if keep:
                accepted.append(cand)
                pbar.update(1)
                pbar.set_postfix(trials=trials, plan_fail=rejected_plan, train_hit=rejected_train)
            else:
                rejected_plan += 1
                pbar.set_postfix(trials=trials, plan_fail=rejected_plan, train_hit=rejected_train)

        pbar.close()
    finally:
        for e in envs.values():
            try:
                e.close()
            except Exception:
                pass

    if len(accepted) < n_seeds:
        raise RuntimeError(
            f"only found {len(accepted)}/{n_seeds} solvable seeds after {trials} trials "
            f"(train_pool={len(train_seeds)}, plan_fail={rejected_plan})"
        )
    return accepted


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to eval/configs/<task>.yaml")
    p.add_argument("--n-seeds", type=int, default=None, help="Override YAML seed_select.n_seeds")
    p.add_argument("--rng-seed", type=int, default=0, help="RNG seed for candidate sampling")
    p.add_argument("--out", default=None, help="Output JSON path (required for saving; no default in competition release)")
    p.add_argument("--max-trials", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_task_config(args.config)
    sel = cfg.get("seed_select") or {}
    n_seeds = int(args.n_seeds if args.n_seeds is not None else sel.get("n_seeds", 20))
    seed_range = tuple(sel.get("seed_range") or [0, 1_000_000])

    print(f"task={cfg.get('task_id')} env={cfg.get('env_id')}")
    train_seeds = load_train_seeds(cfg.get("train_demo_jsons") or [])
    print(f"training seed pool size: {len(train_seeds)}")
    print(f"variants to validate: {(sel.get('check_variants') or list((cfg.get('variants') or {}).keys()))}")

    seeds = select_seeds(
        cfg,
        n_seeds=n_seeds,
        seed_range=seed_range,
        rng_seed=args.rng_seed,
        verbose=args.verbose,
        max_trials=args.max_trials,
    )

    out = args.out or f"/tmp/{cfg.get('task_id', 'task')}_seeds.json"
    path = save_seed_list(
        out,
        seeds,
        meta={
            "task_id": cfg.get("task_id"),
            "env_id": cfg.get("env_id"),
            "config": str(resolve_path(args.config)),
            "n_seeds": n_seeds,
            "rng_seed": args.rng_seed,
            "train_seed_pool_size": len(train_seeds),
            "check_variants": sel.get("check_variants"),
            "require_all_variants": sel.get("require_all_variants", True),
            "disjoint_from_train": True,
        },
    )
    print(f"wrote {len(seeds)} seeds → {path}")
    print(f"seeds: {seeds}")


if __name__ == "__main__":
    # Avoid OpenBLAS / Vulkan thread fights on shared nodes.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
