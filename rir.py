# rir.py
from __future__ import annotations
from typing import Tuple, Dict
import numpy as np
import pyroomacoustics as pra

def _room_absorption_from_rt60(rt60_s: float, dims: Tuple[float, float, float], model: str = "Eyring") -> float:
    L, W, H = dims
    volume = L * W * H
    area = 2.0 * (L * W + L * H + W * H)
    if rt60_s <= 0.0:
        return 0.2  # sane fallback
    if model.lower().startswith("eyr"):
        # Eyring inverse: alpha = 1 - exp(-0.161 V / (A * RT60))
        alpha = 1.0 - np.exp(-0.161 * volume / (area * rt60_s))
    else:
        # Sabine inverse: alpha = 0.161 V / (A * RT60)
        alpha = 0.161 * volume / (area * rt60_s)
    # clamp to (0, 1)
    return float(np.clip(alpha, 1e-4, 0.99))

def clamp_rir_len(rt60_s: float, rir_len_max_s: float) -> float:
    return float(min(1.3 * rt60_s, rir_len_max_s))

def _compute_max_order_default(
    dims: Tuple[float, float, float], cfg: Dict, rir_len_s: float
) -> int:
    """
    If cfg['sim']['max_order_default'] is:
      - int -> return it
      - 'auto' (or missing) -> estimate from path length & room size, with a cap
    """
    sim = cfg.get("sim", {})
    val = sim.get("max_order_default", "auto")
    cap = int(sim.get("max_order_cap", 20))
    if isinstance(val, (int, np.integer, float)) and not isinstance(val, bool):
        return int(max(0, min(int(val), cap)))

    # AUTO heuristic:
    # Max useful path length ~ c * rir_len_s.
    # Roughly, each extra order adds ~ one more wall encounter.
    # A conservative proxy: order ≈ ceil( path_len / (2 * min_dim) ) + 1
    c = float(sim.get("c_mps", 343.0))
    path_len = c * float(rir_len_s)
    min_dim = max(1e-3, float(min(dims)))
    est = int(np.ceil(path_len / (2.0 * min_dim))) + 1
    return int(max(1, min(est, cap)))

def generate_rir_pra(
    room_spec: Tuple[float, float, float, float, str],
    mic_xyz_room: np.ndarray,
    src_xyz_room: np.ndarray,
    fs_hz: int,
    rir_len_s: float,
    cfg: Dict,
) -> np.ndarray:
    """
    room_spec: (L, W, H, rt60_s, model) with model in {"Eyring","Sabine"}
    Returns: array [M, N]
    """
    L, W, H, rt60_s, model = room_spec
    dims = (float(L), float(W), float(H))
    fs = int(fs_hz)

    alpha = _room_absorption_from_rt60(float(rt60_s), dims, model=model)
    max_order = _compute_max_order_default(dims, cfg, rir_len_s)

    # Build room
    room = pra.ShoeBox(
        dims,
        fs=fs,
        materials=pra.Material(alpha),
        max_order=max_order,
        air_absorption=True,
        ray_tracing=False,
    )

    # Add mics (one per row)
    M = mic_xyz_room.shape[0]
    room.add_microphone_array(mic_xyz_room.T)

    # Add single source
    room.add_source(src_xyz_room, signal=np.zeros(1))  # no signal; we only need IRs


    # Compute RIRs
    room.compute_rir()
    # Gather and pad/trim to target length
    n = int(np.round(rir_len_s * fs))
    rirs = []
    for m in range(M):
        ir = np.array(room.rir[m][0], dtype=np.float32)  # source 0
        if len(ir) < n:
            pad = np.zeros(n - len(ir), dtype=np.float32)
            ir = np.concatenate([ir, pad], axis=0)
        else:
            ir = ir[:n]
        rirs.append(ir)
    return np.stack(rirs, axis=0)
