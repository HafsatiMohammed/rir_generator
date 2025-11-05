# samplers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import yaml
from geometry import Pose, distance_interval_for_height_and_az



def _polar_in_disk(rng: np.random.Generator, radius: float) -> Tuple[float, float]:
    # uniform over area: r ~ sqrt(U)*R, theta ~ U(0, 2pi)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0))
    th = rng.uniform(0.0, 2.0 * np.pi)
    return float(r * np.cos(th)), float(r * np.sin(th))

@dataclass(frozen=True)
class RoomSpec:
    L: float
    W: float
    H: float
    rt60_s: float
    model: str  # "Eyring" or "Sabine"

def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _rng_choice_with_p(rng: np.random.Generator, items: List, p: List[float]):
    idx = rng.choice(len(items), p=np.array(p) / np.sum(p))
    return items[idx]

def _uniform_range(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))

def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))

def rt60_bin_schedule(split: str, cfg: dict) -> List[Tuple[Tuple[float,float], int]]:
    bins = cfg["rt60"]["bins_s"]
    counts = cfg["rt60"]["counts"][split]
    return list(zip([tuple(b) for b in bins], [int(c) for c in counts]))

def sample_rt60_in_bin(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))

def pick_room_family(rt60_s: float, cfg: dict, rng: np.random.Generator) -> str:
    for grp in cfg["rooms"]["pairing"]:
        lo, hi = grp["span"]
        if (rt60_s >= lo) and (rt60_s < hi):
            families = ["small", "medium", "large"]
            weights = [grp["small"], grp["medium"], grp["large"]]
            return _rng_choice_with_p(rng, families, weights)
    return "medium"

def sample_room_size(rt60_s: float, rng: np.random.Generator, cfg: dict) -> Tuple[float,float,float]:
    fam = pick_room_family(rt60_s, cfg, rng)
    spec = cfg["rooms"][fam]
    L = _uniform_range(rng, *spec["L_range"])
    W = _uniform_range(rng, *spec["W_range"])
    H = _uniform_range(rng, *spec["H_range"])
    return float(L), float(W), float(H)

def sample_array_pose(rng: np.random.Generator, room_dims: Tuple[float,float,float], cfg: dict) -> Pose:
    L, W, H = room_dims
    clear = cfg["sim"]["clearance_m"]
    cx = _uniform_range(rng, clear, L - clear)
    cy = _uniform_range(rng, clear, W - clear)
    if cfg["array"].get("center_near_room_center", False):
        # sample within a disk of radius center_radius_m around (L/2, W/2), then clamp to keep clearance
        rc = float(cfg["array"].get("center_radius_m", 0.5))
        dx, dy = _polar_in_disk(rng, rc)
        cx = np.clip(L * 0.5 + dx, clear, L - clear)
        cy = np.clip(W * 0.5 + dy, clear, W - clear)
    else:
        cx = _uniform_range(rng, clear, L - clear)
        cy = _uniform_range(rng, clear, W - clear)


    z_lo, z_hi = cfg["array"]["center_height_range_m"]
    z_lo = max(z_lo, clear)
    z_hi = min(z_hi, H - clear)
    cz = _uniform_range(rng, z_lo, z_hi)
    yaw = _uniform_range(rng, *cfg["array"]["yaw_deg_range"])
    mixtures = cfg["array"]["pitch_tilt_deg_mixture"]
    buckets = [m["range"] for m in mixtures]
    probs = [m["p"] for m in mixtures]
    r = _rng_choice_with_p(rng, list(range(len(buckets))), probs)
    pitch = _uniform_range(rng, buckets[r][0], buckets[r][1])
    pitch += _uniform_range(rng, -cfg["array"]["pitch_tilt_jitter_deg"], cfg["array"]["pitch_tilt_jitter_deg"])
    pitch = float(np.clip(pitch, 0.0, 90.0))
    roll = float(rng.normal(0.0, cfg["array"]["roll_deg_sigma"]))
    return Pose(roll_deg=roll, pitch_deg=pitch, yaw_deg=yaw, center_m=np.array([cx, cy, cz], dtype=float))

def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

def sample_distance_and_height(
    rng: np.random.Generator,
    room_dims: Tuple[float,float,float],
    pose: Pose,
    cfg: dict,
    use_bin_prob: bool = False,
) -> Tuple[float, float]:
    d_lo, d_hi = cfg["distances"]["range_m"]
    if use_bin_prob:
        bins = cfg["distances"]["bins"]
        probs = [b["p"] for b in bins]
        bidx = _rng_choice_with_p(rng, list(range(len(bins))), probs)
        lo, hi = bins[bidx]["range"]
        d = _log_uniform(rng, lo, hi)
    else:
        d = _log_uniform(rng, d_lo, d_hi)
    radial_max = cfg["distances"]["radial_frac_of_min_wh"] * min(room_dims[0], room_dims[1])
    d = min(d, radial_max)
    mu, sigma = cfg["heights"]["speech_mu_m"], cfg["heights"]["speech_sigma_m"]
    lo, hi = cfg["heights"]["clip_range_m"]
    h_s = float(np.clip(rng.normal(mu, sigma), lo, hi))
    clear = cfg["sim"]["clearance_m"]
    h_s = float(np.clip(h_s, clear, room_dims[2] - clear))
    return d, h_s

# ---------- New: Feasible sampler for (d, h_s) given azimuth ----------

def sample_distance_and_height_feasible(
    rng: np.random.Generator,
    room_dims: Tuple[float,float,float],
    pose: Pose,
    cfg: dict,
    az_deg: float,
    max_height_tries: int = 64,
) -> Tuple[float, float, Tuple[float,float]]:
    """
    Sample h_s first, then derive a guaranteed-feasible distance interval [d_min, d_max].
    Finally sample d log-uniform in that interval.
    Returns (d, h_s, (d_min, d_max)). Raises RuntimeError if no feasible pair found.
    """
    d_lo_cfg, d_hi_cfg = cfg["distances"]["range_m"]
    radial_max = cfg["distances"]["radial_frac_of_min_wh"] * min(room_dims[0], room_dims[1])
    d_hi_cfg = min(d_hi_cfg, radial_max)
    clear = cfg["sim"]["clearance_m"]

    mu, sigma = cfg["heights"]["speech_mu_m"], cfg["heights"]["speech_sigma_m"]
    h_lo, h_hi = cfg["heights"]["clip_range_m"]
    h_lo = max(h_lo, clear)
    h_hi = min(h_hi, room_dims[2] - clear)

    for _ in range(max_height_tries):
        h_s = float(np.clip(rng.normal(mu, sigma), h_lo, h_hi))
        interval = distance_interval_for_height_and_az(
            az_deg=az_deg, h_s_m=h_s, pose=pose,
            room_dims=room_dims, clearance_m=clear,
            d_lo_cfg=d_lo_cfg, d_hi_cfg=d_hi_cfg,
        )
        if interval is None:
            continue
        d_min, d_max = interval
        if not (d_max > d_min):
            continue
            
        prior = str(cfg["distances"].get("prior", "log_uniform_feasible")).lower()
        if prior == "uniform_feasible":
            d = float(rng.uniform(d_min, d_max))
        elif prior == "power_feasible":
            # p(d) ∝ 1/d^k on [d_min, d_max], k in [0,1). k=0 → uniform; k→1- still mild small-d bias
            k = float(cfg["distances"].get("power_k", 0.3))
            if abs(k) < 1e-9:
                d = float(rng.uniform(d_min, d_max))
            else:
                # Invert CDF of ∫ d^{-k} dd ⇒ d^{1-k}
                u = rng.uniform(0.0, 1.0)
                a = d_min ** (1.0 - k)
                b = d_max ** (1.0 - k)
                d = float((a + u * (b - a)) ** (1.0 / (1.0 - k)))
        else:  # default: log-uniform (previous behavior)
            d = float(np.exp(rng.uniform(np.log(d_min), np.log(d_max))))
        return d, h_s, (d_min, d_max)

    raise RuntimeError("No feasible (d, h_s) found for this azimuth and pose.")
