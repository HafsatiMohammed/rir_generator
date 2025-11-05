# array.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np

from geometry import Pose
from samplers import sample_array_pose

DEFAULT_UCA4_RADIUS_M = 0.0277  # 27.7 mm per spec

@dataclass(frozen=True)
class JitterParams:
    clean_room: bool
    chosen_sigma_m: float
    clip_m: float
    which_bucket: int  # 0,1,2 per mixture

def mic_positions_uca4(radius_m: float) -> np.ndarray:
    """
    Base UCA-4 coordinates in the array frame (z=0 plane), Mic0 on +x.
    Returns shape (4,3).
    """
    R = float(radius_m)
    base = np.array([
        [ R, 0.0, 0.0],   # Mic 0 (A)
        [ 0.0, R, 0.0],   # Mic 1 (B)
        [-R, 0.0, 0.0],   # Mic 2 (C)
        [ 0.0,-R, 0.0],   # Mic 3 (D)
    ], dtype=float)
    return base

def _rng_choice_idx(rng: np.random.Generator, probs: np.ndarray) -> int:
    probs = np.maximum(probs, 0.0)
    s = probs.sum()
    if s <= 0:
        probs = np.array([1.0, 0.0, 0.0])
    else:
        probs = probs / s
    return int(rng.choice(len(probs), p=probs))

def _sample_jitter_params(rng: np.random.Generator, cfg: Dict) -> JitterParams:
    """Per-room geometry jitter mixture with clean-room option."""
    arr_cfg = cfg.get("array", {})
    clean_p = float(arr_cfg.get("clean_room_probability", 0.08))  # default 8%
    if rng.uniform() < clean_p:
        return JitterParams(clean_room=True, chosen_sigma_m=0.0, clip_m=0.0, which_bucket=-1)

    mix = arr_cfg.get("jitter_mixture", None)
    if mix is None:
        # sensible defaults per spec §7
        mix = [
            {"p": 0.80, "sigma_mm": 2.0, "clip_mm": 5.0},
            {"p": 0.15, "sigma_mm": 4.0, "clip_mm": 8.0},
            {"p": 0.05, "sigma_mm_range_mm": [6.0, 8.0], "clip_mm": 8.0},
        ]

    probs = np.array([float(m.get("p", 0.0)) for m in mix], dtype=float)
    k = _rng_choice_idx(rng, probs)
    entry = mix[k]

    if "sigma_mm" in entry:
        sigma_mm = float(entry["sigma_mm"])
    else:
        lo, hi = [float(x) for x in entry.get("sigma_mm_range_mm", [6.0, 8.0])]
        sigma_mm = float(rng.uniform(lo, hi))

    clip_mm = float(entry.get("clip_mm", max(5.0, 3.0 * sigma_mm)))
    return JitterParams(
        clean_room=False,
        chosen_sigma_m=sigma_mm / 1000.0,
        clip_m=clip_mm / 1000.0,
        which_bucket=k,
    )

def _apply_per_mic_jitter(
    rng: np.random.Generator, base_xyz_arr: np.ndarray, jp: JitterParams
) -> np.ndarray:
    """Apply iid 3D jitter in the array frame, clipped component-wise."""
    if jp.clean_room or jp.chosen_sigma_m <= 0.0:
        return base_xyz_arr.copy()
    M = base_xyz_arr.shape[0]
    jitter = rng.normal(loc=0.0, scale=jp.chosen_sigma_m, size=(M, 3))
    jitter = np.clip(jitter, -jp.clip_m, jp.clip_m)
    return base_xyz_arr + jitter

def sample_room_array_geometry(
    rng: np.random.Generator, cfg: Dict, room_dims: Tuple[float, float, float]
) -> Tuple[Pose, np.ndarray, np.ndarray, JitterParams]:
    """
    Sample array pose and per-mic geometry jitter ONCE per room
    and return:
      - pose (room frame)
      - mic_xyz_array_m (after jitter, array frame)
      - mic_xyz_room_m  (posed to room)
      - jitter params (for metadata)
    """
    # 1) Pose
    pose = sample_array_pose(rng, room_dims, cfg)

    # 2) Base UCA-4 mics (array frame)
    radius_m = float(cfg.get("array", {}).get("radius_m", DEFAULT_UCA4_RADIUS_M))
    base = mic_positions_uca4(radius_m)  # (4,3) in array frame

    # 3) Per-room jitter params & apply jitter in array frame
    jp = _sample_jitter_params(rng, cfg)
    mic_xyz_array_m = _apply_per_mic_jitter(rng, base, jp)

    # 4) Pose mics to room: x_room = center + R_ar @ x_array
    R_ar = pose.R_array_to_room()
    mic_xyz_room_m = (R_ar @ mic_xyz_array_m.T).T + pose.center_m.reshape(1, 3)

    return pose, mic_xyz_array_m, mic_xyz_room_m, jp
