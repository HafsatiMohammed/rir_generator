# io_meta.py
from __future__ import annotations
from dataclasses import asdict
from typing import Dict, Tuple
import json
import os
import numpy as np
import soundfile as sf

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def write_room_meta(
    out_dir: str,
    room_id: str,
    room_dims: Tuple[float,float,float],
    rt60_s: float,
    array_id: str,
    pose_room: Dict[str, float],
    mic_xyz_arr_m: np.ndarray,
    mic_xyz_room_m: np.ndarray,
    geom_jitter_mm: np.ndarray,
    sim_defaults: Dict[str, float],
    model: str = "Eyring",
) -> str:
    _ensure_dir(out_dir)
    meta = {
        "room": {
            "id": room_id,
            "dims_m": list(map(float, room_dims)),
            "rt60_s": float(rt60_s),
            "model": model,
        },
        "array": {
            "id": array_id,
            "pose_room": {
                "roll_deg": float(pose_room["roll_deg"]),
                "pitch_deg": float(pose_room["pitch_deg"]),
                "yaw_deg": float(pose_room["yaw_deg"]),
                "center_m": list(map(float, pose_room["center_m"])),
            },
            "mic_xyz_array_m": mic_xyz_arr_m.tolist(),
            "mic_xyz_room_m": mic_xyz_room_m.tolist(),
            "geom_jitter_mm": geom_jitter_mm.tolist(),
            "clean_room": bool(np.allclose(geom_jitter_mm, 0.0)),
        },
        "sim_defaults": sim_defaults,
    }
    fp = os.path.join(out_dir, "room_meta.json")
    with open(fp, "w") as f:
        json.dump(meta, f, indent=2)
    return fp

def make_pose_dir(
    base_out: str,
    split: str,
    room_id: str,
    array_id: str,
    rt60_ms: int,
    srcpose_idx: int,
) -> str:
    d = os.path.join(
        base_out, split, f"room-{room_id}", f"array-{array_id}",
        f"rt60-{rt60_ms}"
    )
    d = os.path.join(d, f"srcpose-{srcpose_idx:04d}")
    _ensure_dir(d)
    return d

def rir_filename(
    room_id: str, rt60_ms: int, rdeg: float,
    az_deg: float, el_deg: float, d_cm: float, fs: int, rir_ms: int,
    seed: int, mic_index: int,
) -> str:
    return (
        f"rir_room-{room_id}_rt60-{rt60_ms}_rot-{int(round(rdeg))}_"
        f"az-{int(round(az_deg))}_el-{int(round(el_deg))}_"
        f"d-{int(round(d_cm))}_fs-{fs}_len-{rir_ms}_seed-{seed}_mic-{mic_index}.wav"
    )

def write_pose_meta(
    pose_dir: str,
    room_meta_rel: str,
    unit_vec_array: np.ndarray,
    az_deg: float,
    el_deg: float,
    distance_m: float,
    src_xyz_room_m: np.ndarray,
    method: str,
    rir_len_s: float,
    seed: int,
):
    meta = {
        "room_meta": room_meta_rel,
        "src_pose": {
            "direction_unit_xyz_array": unit_vec_array.tolist(),
            "azimuth_deg": float(az_deg),
            "elevation_deg": float(el_deg),
            "distance_m": float(distance_m),
            "src_xyz_room_m": src_xyz_room_m.tolist(),
        },
        "sim": {"method": method, "rir_len_s": float(rir_len_s), "seed": int(seed)}
    }
    fp = os.path.join(pose_dir, "meta.json")
    with open(fp, "w") as f:
        json.dump(meta, f, indent=2)
    return fp

def save_wavs(
    pose_dir: str,
    base_filename: str,
    rirs: np.ndarray,  # (M, N)
    fs: int,
):
    M = rirs.shape[0]
    for m in range(M):
        fp = os.path.join(pose_dir, base_filename.replace("mic-0", f"mic-{m}"))
        sf.write(fp, rirs[m], fs, subtype="FLOAT")
