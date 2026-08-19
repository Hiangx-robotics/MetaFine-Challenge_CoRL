"""Shared helpers for per-task MetaFine policy evaluation.

Task-specific scripts under ``eval/`` own CLI + scoring; this module owns the
plumbing that every task reuses:

* load a task YAML config + training-demo seed pool
* build ``sensor_configs`` that match the training replay metadata
* load a LeRobot PI0 checkpoint
* convert ManiSkill obs → PI0 batch
* run one closed-loop episode
* build Perception DR profiles (camera pos+rot joint; light separate)
* trapezoidal AUSC on a success-rate curve
"""

from __future__ import annotations

import json
import os
import os.path as osp
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Config / seed I/O                                                           #
# --------------------------------------------------------------------------- #

def load_task_config(path: str | Path) -> dict:
    """Load a task eval YAML (see ``eval/configs/*.yaml``)."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"task config must be a mapping: {path}")
    cfg["_config_path"] = str(path.resolve())
    return cfg


def resolve_path(p: str | Path, *, root: Path = REPO_ROOT) -> Path:
    """Resolve a path relative to the repo root when not absolute."""
    p = Path(p)
    return p if p.is_absolute() else (root / p)


def load_train_seeds(demo_jsons: Sequence[str | Path]) -> set[int]:
    """Collect ``episode_seed`` values from ManiSkill trajectory JSON sidecars."""
    seeds: set[int] = set()
    for p in demo_jsons:
        path = resolve_path(p)
        if not path.exists():
            raise FileNotFoundError(f"train demo json not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        episodes = data.get("episodes", [])
        # Some sidecars nest under env_info; episodes is always top-level here.
        for ep in episodes:
            s = ep.get("episode_seed", ep.get("seed"))
            if s is not None:
                seeds.add(int(s))
    return seeds


def load_seed_list(path: str | Path) -> List[int]:
    """Load a previously saved eval seed list JSON.

    Accepted shapes:
      ``{"seeds": [..]}``  or  a bare ``[..]`` list.
    """
    path = resolve_path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [int(s) for s in data]
    if isinstance(data, dict) and "seeds" in data:
        return [int(s) for s in data["seeds"]]
    raise ValueError(f"unrecognised seed list format: {path}")


def save_seed_list(
    path: str | Path,
    seeds: Sequence[int],
    *,
    meta: Optional[dict] = None,
) -> Path:
    """Persist an eval seed list with optional provenance metadata."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seeds": [int(s) for s in seeds]}
    if meta:
        payload["meta"] = meta
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Env construction                                                            #
# --------------------------------------------------------------------------- #

def sensor_configs_from_task(cfg: dict) -> dict:
    """Build ``sensor_configs`` kwargs matching the task YAML / training data."""
    sensor = cfg.get("sensor") or {}
    sc: dict[str, Any] = {
        "shader_pack": sensor.get("shader_pack", "default"),
        "width": int(sensor.get("width", 512)),
        "height": int(sensor.get("height", 512)),
    }
    fov = sensor.get("base_camera_fov")
    if fov is not None:
        sc["base_camera"] = {"fov": float(fov)}
    return sc


def build_base_env_kwargs(cfg: dict, *, extra: Optional[dict] = None) -> dict:
    """Common ``gym.make`` kwargs for a task config (no DR, no variant)."""
    kwargs: dict[str, Any] = {
        "obs_mode": cfg.get("obs_mode", "rgb"),
        "control_mode": cfg.get("control_mode", "pd_joint_delta_pos"),
        "render_mode": "rgb_array",
        "sensor_configs": sensor_configs_from_task(cfg),
        "human_render_camera_configs": dict(
            shader_pack=(cfg.get("sensor") or {}).get("shader_pack", "default")
        ),
        "viewer_camera_configs": dict(
            shader_pack=(cfg.get("sensor") or {}).get("shader_pack", "default")
        ),
        "sim_backend": cfg.get("sim_backend", "physx_cpu"),
    }
    if cfg.get("max_episode_steps") is not None:
        kwargs["max_episode_steps"] = int(cfg["max_episode_steps"])
    if cfg.get("object_name") is not None:
        kwargs["object_name"] = cfg["object_name"]
    if extra:
        kwargs.update(extra)
    return kwargs


def build_perception_profiles(base_env_kwargs: dict, cfg: dict) -> Dict[str, dict]:
    """Return ``{clean, cam_l1..3, light_l1..3}`` env-kwarg profiles.

    Camera levels apply position + rotation jitter jointly (one AUSC axis).
    Light levels widen the ambient band around 0.5 (second AUSC axis).
    """
    perc = cfg.get("perception") or {}
    profiles: Dict[str, dict] = {"clean": dict(base_env_kwargs)}

    cam_pos = list(perc.get("camera_pos_levels") or [0.03, 0.06, 0.12])
    cam_rot = list(perc.get("camera_rot_levels_deg") or [2.0, 6.0, 12.0])
    n = min(len(cam_pos), len(cam_rot))
    for i in range(n):
        profiles[f"cam_l{i+1}"] = {
            **base_env_kwargs,
            "eval_randomize_camera": True,
            "eval_camera_pos_jitter": float(cam_pos[i]),
            "eval_camera_rot_jitter_deg": float(cam_rot[i]),
            "eval_randomize_light": False,
        }

    light_deltas = list(perc.get("light_ambient_delta_levels") or [0.10, 0.25, 0.40])
    for i, delta in enumerate(light_deltas):
        delta = max(0.0, float(delta))
        profiles[f"light_l{i+1}"] = {
            **base_env_kwargs,
            "eval_randomize_light": True,
            "eval_ambient_low": max(0.0, 0.5 - delta),
            "eval_ambient_high": min(1.0, 0.5 + delta),
            "eval_randomize_camera": False,
        }
    return profiles


def compute_ausc(values: Sequence[float]) -> float:
    """Trapezoidal area under a success curve mapped onto x∈[0,1]."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size < 2:
        return float("nan")
    x = np.linspace(0.0, 1.0, num=arr.size)
    return float(np.trapz(arr, x))


def summarise_perception(profile_results: Dict[str, dict]) -> dict:
    """Attach ``ausc_camera`` / ``ausc_light`` / ``ausc_mean`` to profile results."""
    out = dict(profile_results)
    ausc_values: list[float] = []

    cam_names = ["clean"] + sorted(
        [k for k in profile_results if k.startswith("cam_l")],
        key=lambda s: int(s.split("l")[-1]),
    )
    if len(cam_names) >= 2 and all(n in profile_results for n in cam_names):
        srs = [profile_results[n]["success_rate"] for n in cam_names]
        cam_ausc = compute_ausc(srs)
        out["ausc_camera"] = {"value": cam_ausc, "curve": dict(zip(cam_names, srs))}
        if np.isfinite(cam_ausc):
            ausc_values.append(float(cam_ausc))

    light_names = ["clean"] + sorted(
        [k for k in profile_results if k.startswith("light_l")],
        key=lambda s: int(s.split("l")[-1]),
    )
    if len(light_names) >= 2 and all(n in profile_results for n in light_names):
        srs = [profile_results[n]["success_rate"] for n in light_names]
        light_ausc = compute_ausc(srs)
        out["ausc_light"] = {"value": light_ausc, "curve": dict(zip(light_names, srs))}
        if np.isfinite(light_ausc):
            ausc_values.append(float(light_ausc))

    if ausc_values:
        out["ausc_mean"] = {"value": float(np.mean(ausc_values))}
    return out


# --------------------------------------------------------------------------- #
# PI0 load / obs / rollout                                                    #
# --------------------------------------------------------------------------- #

def load_pi0_policy(
    policy_path: str,
    device: str = "cuda",
    tokenizer_path: str | None = None,
):
    """Load PI0Policy + preprocessor/postprocessor from a pretrained_model dir."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy

    if tokenizer_path:
        os.environ["LEROBOT_TOKENIZER_PATH"] = tokenizer_path

    cfg = PreTrainedConfig.from_pretrained(policy_path)
    cfg.device = device
    policy = PI0Policy.from_pretrained(policy_path, config=cfg)
    policy.to(device)
    policy.eval()

    # Also mirror pi0/evaluate.py: repair PaliGemma tied embed_tokens ↔ lm_head.
    try:
        pg = policy.model.paligemma_with_expert.paligemma
        if not torch.equal(
            pg.lm_head.weight.data,
            pg.model.language_model.embed_tokens.weight.data,
        ):
            pg.model.language_model.embed_tokens.weight = pg.lm_head.weight
    except AttributeError:
        pass

    overrides: dict = {"device_processor": {"device": device}}
    if tokenizer_path:
        # TokenizerProcessorStep takes tokenizer_name (HF id or local path).
        overrides["tokenizer_processor"] = {"tokenizer_name": tokenizer_path}

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=policy_path,
        preprocessor_overrides=overrides,
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, preprocessor, postprocessor


def maniskill_obs_to_batch(obs: dict, task: str) -> dict:
    """Map ManiSkill rgb obs → LeRobot PI0 batch (pre-rename camera keys)."""
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE

    qpos = obs["agent"]["qpos"]
    if isinstance(qpos, torch.Tensor):
        qpos = qpos.detach().cpu()
    state = torch.as_tensor(np.asarray(qpos, dtype=np.float32))

    images: dict[str, torch.Tensor] = {}
    for cam_name in ("base_camera", "hand_camera"):
        rgb = obs["sensor_data"][cam_name]["rgb"]
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu()
        rgb_np = np.asarray(rgb, dtype=np.uint8)
        if rgb_np.ndim == 3:
            rgb_np = rgb_np[np.newaxis]
        img_t = torch.from_numpy(rgb_np).permute(0, 3, 1, 2).contiguous().float() / 255.0
        images[f"{OBS_IMAGES}.{cam_name}"] = img_t

    return {OBS_STATE: state, "task": task, **images}


def adapt_action(action: np.ndarray, env_dim: int, default_last: float = 0.0) -> np.ndarray:
    """Pad/truncate policy action to ``env.action_space`` dim."""
    policy_dim = action.shape[-1]
    if policy_dim == env_dim:
        return action
    if policy_dim == env_dim - 1:
        pad = np.full((*action.shape[:-1], 1), default_last, dtype=action.dtype)
        return np.concatenate([action, pad], axis=-1)
    if policy_dim > env_dim:
        return action[..., :env_dim]
    raise ValueError(f"action dim mismatch: policy={policy_dim}, env={env_dim}")


def _extract_success(info: Any) -> bool:
    """Pull a boolean success from ManiSkill / CPUGymWrapper info dicts."""
    if not isinstance(info, dict):
        return False
    if "final_info" in info:
        fi = info["final_info"]
        if isinstance(fi, dict):
            ep = fi.get("episode") or fi
            for key in ("success_at_end", "success"):
                if key in ep:
                    v = ep[key]
                    if isinstance(v, torch.Tensor):
                        v = v.detach().cpu().numpy()
                    if isinstance(v, np.ndarray):
                        v = bool(np.asarray(v).reshape(-1)[0]) if v.size else False
                    return bool(v)
            if "success" in fi:
                v = fi["success"]
                if isinstance(v, torch.Tensor):
                    v = v.item() if v.numel() == 1 else bool(v.detach().cpu().numpy().reshape(-1)[0])
                return bool(v)
    if "success" in info:
        v = info["success"]
        if isinstance(v, torch.Tensor):
            v = v.item() if v.numel() == 1 else bool(v.detach().cpu().numpy().reshape(-1)[0])
        return bool(v)
    return False


def run_episode(
    env,
    policy,
    preprocessor,
    postprocessor,
    *,
    task: str,
    seed: int,
    max_steps: int,
    default_last_action: float = 0.0,
    on_step: Optional[Callable[[int, dict, Any], None]] = None,
) -> dict:
    """Roll out one episode; return success / length / actions / extras.

    ``env`` should already be wrapped with ``CPUGymWrapper`` (callers own the
    env lifecycle — this helper never closes it). ``on_step(step_idx, obs,
    info)`` is called after every ``env.step`` for task-specific harvesting.
    """
    policy.reset()
    obs, info = env.reset(seed=int(seed))
    actions: list[np.ndarray] = []
    last_info = info
    extras: dict[str, Any] = {}

    env_dim = int(np.prod(env.action_space.shape))
    for t in range(int(max_steps)):
        batch = maniskill_obs_to_batch(obs, task)
        batch = preprocessor(batch)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action = postprocessor(action)
        act_np = np.asarray(action, dtype=np.float32)
        if act_np.ndim == 1:
            act_np = act_np[None, :]
        act_np = adapt_action(act_np, env_dim, default_last_action)
        obs, _rew, _term, _trunc, info = env.step(act_np[0])
        last_info = info
        actions.append(act_np[0].copy())
        if on_step is not None:
            on_step(t, obs, info)
        if _extract_success(info):
            break

    success = _extract_success(last_info)
    try:
        base = env.unwrapped
        ev = base.evaluate()
        if isinstance(ev, dict) and "success" in ev:
            s = ev["success"]
            if isinstance(s, torch.Tensor):
                success = bool(s.detach().cpu().numpy().reshape(-1)[0])
            else:
                success = bool(s)
            extras["evaluate"] = {
                k: (v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v)
                for k, v in ev.items()
                if k != "success"
            }
        if hasattr(base, "grasped_part"):
            extras["grasped_part"] = base.grasped_part
        if hasattr(base, "stage_id"):
            sid = base.stage_id
            if isinstance(sid, torch.Tensor):
                extras["stage_id"] = int(sid.detach().cpu().numpy().reshape(-1)[0])
            else:
                extras["stage_id"] = int(sid)
    except Exception:
        pass

    return {
        "seed": int(seed),
        "success": bool(success),
        "episode_length": len(actions),
        "actions": actions,
        "extras": extras,
    }


def dump_json(path: str | Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Video helpers (policy-view RGB from sensor_data)                             #
# --------------------------------------------------------------------------- #

def _rgb_to_uint8(rgb: Any) -> np.ndarray:
    """Convert a ManiSkill sensor RGB tensor/array to HxWx3 uint8."""
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    arr = np.asarray(rgb)
    # Drop leading batch dim if present: (1, H, W, C) → (H, W, C)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected HxWxC rgb, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        if arr.max() <= 1.0 + 1e-3:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    return arr


def obs_to_frame(obs: dict, cams: Sequence[str] = ("base_camera", "hand_camera")) -> np.ndarray:
    """Side-by-side RGB frame from the cameras the policy actually sees.

    Pads heights when cameras differ in resolution so they can be concat'd.
    """
    panels: list[np.ndarray] = []
    sensor = obs.get("sensor_data") or {}
    for cam in cams:
        if cam not in sensor or "rgb" not in sensor[cam]:
            continue
        panels.append(_rgb_to_uint8(sensor[cam]["rgb"]))
    if not panels:
        raise KeyError("obs has no usable camera rgb under sensor_data")
    if len(panels) == 1:
        return panels[0]
    h = max(p.shape[0] for p in panels)
    padded = []
    for p in panels:
        if p.shape[0] < h:
            pad = np.zeros((h - p.shape[0], p.shape[1], p.shape[2]), dtype=p.dtype)
            p = np.concatenate([p, pad], axis=0)
        padded.append(p)
    return np.concatenate(padded, axis=1)


def save_rgb_video(frames: Sequence[np.ndarray], path: str | Path, fps: int = 30) -> Path:
    """Write a list of HxWx3 uint8 frames to an mp4 via imageio."""
    import imageio

    path = Path(path)
    if not frames:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=int(fps))
    try:
        for frame in frames:
            writer.append_data(_rgb_to_uint8(frame))
    finally:
        writer.close()
    return path
