# geometry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

@dataclass(frozen=True)
class Pose:
    """Array pose in the room frame (ZYX extrinsic yaw-pitch-roll)."""
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    center_m: np.ndarray  # shape (3,)

    def R_array_to_room(self) -> np.ndarray:
        """Rotation matrix from array frame to room frame (3x3)."""
        r, p, y = self.roll_deg * DEG2RAD, self.pitch_deg * DEG2RAD, self.yaw_deg * DEG2RAD
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(r), -np.sin(r)],
                       [0, np.sin(r),  np.cos(r)]])
        Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                       [ 0,         1, 0        ],
                       [-np.sin(p), 0, np.cos(p)]])
        Rz = np.array([[np.cos(y), -np.sin(y), 0],
                       [np.sin(y),  np.cos(y), 0],
                       [0,          0,         1]])
        # ZYX extrinsic: yaw(Z) -> pitch(Y) -> roll(X)
        return Rz @ Ry @ Rx

    def R_room_to_array(self) -> np.ndarray:
        return self.R_array_to_room().T


def wrap_deg_0_360(angle_deg: float) -> float:
    a = np.mod(angle_deg, 360.0)
    return a + 360.0 if a < 0 else a


def make_rotated_grid(r_deg: float) -> np.ndarray:
    """
    Global 72 bins: {0,5,...,355}. Local per-room grid = r + 5*k (mod 360).
    """
    base = np.arange(72, dtype=float) * 5.0
    return np.mod(r_deg + base, 360.0)


def quantize_global72(az_deg: float) -> Tuple[int, float]:
    """
    Quantize azimuth to the nearest of the global 72 bins (every 5 degrees).
    Returns: (bin_index [0..71], bin_center_deg)
    """
    az = wrap_deg_0_360(az_deg)
    idx = int(np.round(az / 5.0)) % 72
    return idx, idx * 5.0


def azel_from_unit_u_array(u_arr: np.ndarray) -> Tuple[float, float]:
    """
    Azimuth: angle in array x-y plane from +x (Mic0 axis) toward +y (deg, 0..360)
    Elevation: arcsin(u·z^) (deg, positive above the array plane)
    """
    z_hat = np.array([0.0, 0.0, 1.0])
    u_par = u_arr - np.dot(u_arr, z_hat) * z_hat
    norm_par = np.linalg.norm(u_par)
    if norm_par < 1e-12:
        az = 0.0
    else:
        az = np.arctan2(u_par[1], u_par[0]) * RAD2DEG
        az = wrap_deg_0_360(az)
    el = np.arcsin(np.clip(np.dot(u_arr, z_hat), -1.0, 1.0)) * RAD2DEG
    return az, el


def labels_from_geometry(
    src_xyz_room: np.ndarray,
    array_center_room: np.ndarray,
    pose: Pose,
    near_zenith_min_upar: float = 0.10,
) -> Tuple[np.ndarray, float, float, bool]:
    """
    Compute direction labels in the array frame.
    Returns: (unit_vec_array, az_deg, el_deg, is_near_zenith)
    """
    R_ra = pose.R_room_to_array()
    v_room = src_xyz_room - array_center_room
    v_arr = R_ra @ v_room
    norm = np.linalg.norm(v_arr)
    if norm == 0.0:
        raise ValueError("Source coincides with array center.")
    u_arr = v_arr / norm
    # near-zenith test: ||projection on array plane||
    z_hat = np.array([0.0, 0.0, 1.0])
    u_par = u_arr - np.dot(u_arr, z_hat) * z_hat
    is_near_zenith = (np.linalg.norm(u_par) < near_zenith_min_upar)
    az, el = azel_from_unit_u_array(u_arr)
    return u_arr, az, el, is_near_zenith


def place_source_from_array_center(
    az_deg: float,
    d_m: float,
    h_s_m: float,
    pose: Pose,
) -> np.ndarray:
    """
    Place a source at azimuth (array frame), 3D distance d, and room-frame height h_s.
    Solves for array-frame v = [rho*cos(az), rho*sin(az), v_z] such that:
      dot(v, z_room_in_array) = h_s - center_z
      ||v|| = d
    """
    az = az_deg * DEG2RAD
    R_ar = pose.R_array_to_room()
    R_ra = R_ar.T
    ez_room = np.array([0.0, 0.0, 1.0])
    z_room_in_array = R_ra @ ez_room
    delta_z = h_s_m - pose.center_m[2]

    A = z_room_in_array[0] * np.cos(az) + z_room_in_array[1] * np.sin(az)
    zza = z_room_in_array[2]
    a = (zza * zza + A * A)
    b = -2.0 * A * delta_z
    c = (delta_z * delta_z - (d_m * d_m) * (zza * zza))
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        raise ValueError("Infeasible (d, h_s, pose) combination; resample.")
    sqrt_disc = np.sqrt(disc)
    rho1 = ( -b + sqrt_disc ) / (2.0 * a)
    rho2 = ( -b - sqrt_disc ) / (2.0 * a)
    candidates = [rho for rho in (rho1, rho2) if rho >= 0.0]
    if not candidates:
        raise ValueError("Negative in-plane magnitude solution; resample.")
    rho = max(candidates)
    v_z = (delta_z - rho * A) / (zza if np.abs(zza) > 1e-9 else 1e-9)
    v_arr = np.array([rho * np.cos(az), rho * np.sin(az), v_z])
    # Normalize to exact distance d
    scale = d_m / np.linalg.norm(v_arr)
    v_arr *= scale
    src_xyz_room = pose.center_m + (R_ar @ v_arr)
    return src_xyz_room


# ---------- Feasibility tools for robust sampling ----------

def distance_interval_for_height_and_az(
    az_deg: float,
    h_s_m: float,
    pose: Pose,
    room_dims: Tuple[float, float, float],
    clearance_m: float,
    d_lo_cfg: float,
    d_hi_cfg: float,
    in_bounds_eps: float = 1e-3,
) -> Optional[Tuple[float, float]]:
    """
    Compute a guaranteed-feasible distance interval [d_min, d_max] for a given height and azimuth.
      - d_min from |Δz| ≤ d * ||z_room_in_array||
      - d_max from wall clearances via bisection using the exact placement solver.
    Returns None if no feasible distance exists.
    """
    L, W, H = room_dims
    R_ar = pose.R_array_to_room()
    R_ra = R_ar.T
    z_room_in_array = R_ra @ np.array([0.0, 0.0, 1.0])
    zra = float(np.linalg.norm(z_room_in_array))
    if zra < 1e-9:
        zra = 1.0

    delta_z = float(h_s_m - pose.center_m[2])
    d_min_geom = abs(delta_z) / zra + in_bounds_eps
    d_min = max(d_lo_cfg, d_min_geom)

    def _in_bounds(d: float) -> bool:
        try:
            s = place_source_from_array_center(az_deg, d, h_s_m, pose)
        except Exception:
            return False
        x, y, z = s
        return (clearance_m <= x <= L - clearance_m and
                clearance_m <= y <= W - clearance_m and
                clearance_m <= z <= H - clearance_m)

    # Early rejection
    if d_min > d_hi_cfg:
        return None

    lo = d_min
    hi = d_hi_cfg

    # Ensure lo is valid; if not, increase slightly up to hi
    if not _in_bounds(lo):
        for _ in range(24):
            lo *= 1.1
            if lo >= hi:
                return None
            if _in_bounds(lo):
                break
        else:
            return None

    # Bring hi down to last valid if needed
    if not _in_bounds(hi):
        l, r = lo, hi
        for _ in range(40):
            mid = 0.5 * (l + r)
            if _in_bounds(mid):
                l = mid
            else:
                r = mid
        hi = l

    if hi <= lo:
        return None
    return (lo, hi)
