# cli_generate.py
from __future__ import annotations
import argparse
import hashlib
import json
import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from geometry import Pose, make_rotated_grid, labels_from_geometry, place_source_from_array_center
from samplers import load_config, rt60_bin_schedule, sample_rt60_in_bin, sample_room_size, RoomSpec
from samplers import sample_distance_and_height_feasible
from array import sample_room_array_geometry, mic_positions_uca4, JitterParams
from io_meta import write_room_meta, write_pose_meta, make_pose_dir, rir_filename, save_wavs
from rir import clamp_rir_len, generate_rir_pra

def derive_seed(*ints_or_strs) -> int:
    h = hashlib.sha256()
    for x in ints_or_strs:
        if isinstance(x, (int, np.integer)):
            h.update(int(x).to_bytes(8, "little", signed=False))
        else:
            h.update(str(x).encode("utf-8"))
    return int.from_bytes(h.digest()[:4], "little")

def _room_id(i: int) -> str:
    return f"{i:04d}"

def _absorption_model(rt60_s: float) -> str:
    return "Eyring"

def _validate_inside_room(xyz: np.ndarray, dims: Tuple[float,float,float], clearance: float) -> bool:
    x,y,z = xyz
    L,W,H = dims
    return (clearance <= x <= L - clearance) and (clearance <= y <= W - clearance) and (clearance <= z <= H - clearance)

def main():
    ap = argparse.ArgumentParser(description="Generate a bank of simulated RIRs per spec.")
    ap.add_argument("--split", required=True, choices=["train","val","test"])
    ap.add_argument("--rooms", type=int, required=True)
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--K", type=int, default=36)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config()
    schedule = rt60_bin_schedule(args.split, cfg)
    target_total = sum(c for _, c in schedule)
    if args.rooms != target_total:
        scale = args.rooms / float(target_total)
        new_counts = [max(0, int(round(c * scale))) for _, c in schedule]
        diff = args.rooms - sum(new_counts)
        if diff != 0:
            new_counts[-1] += diff
        schedule = list(zip([b for b,_ in schedule], new_counts))

    out_root = os.path.abspath(args.out)
    os.makedirs(out_root, exist_ok=True)

    global_seed = int(args.seed)
    room_counter = 0
    pbar = tqdm(total=sum(c for _, c in schedule), desc=f"Rooms ({args.split})", unit="room")
    for (bin_lo, bin_hi), count in schedule:
        for _ in range(count):
            room_id = _room_id(room_counter)
            room_seed = derive_seed(global_seed, args.split, room_id)
            rng_room = np.random.default_rng(room_seed)

            rt60_s = sample_rt60_in_bin(rng_room, bin_lo, bin_hi)
            L,W,H = sample_room_size(rt60_s, rng_room, cfg)
            dims = (L,W,H)

            rdeg = float(rng_room.uniform(0.0, 5.0))
            local_grid = make_rotated_grid(rdeg)
            if args.K > 72:
                raise ValueError("K cannot exceed 72.")
            az_set = rng_room.choice(local_grid, size=args.K, replace=False)

            pose, mic_xyz_arr, mic_xyz_room, jp = sample_room_array_geometry(rng_room, cfg, dims)

            room_dir = os.path.join(out_root, args.split, f"room-{room_id}")
            sim_defaults = {"fs_hz": int(args.fs), "c_mps": float(cfg["sim"]["c_mps"])}
            room_meta_path = write_room_meta(
                out_dir=room_dir,
                room_id=f"room_{room_id}",
                room_dims=dims,
                rt60_s=float(rt60_s),
                array_id=cfg["array"]["id"],
                pose_room={
                    "roll_deg": pose.roll_deg,
                    "pitch_deg": pose.pitch_deg,
                    "yaw_deg": pose.yaw_deg,
                    "center_m": pose.center_m.tolist(),
                },
                mic_xyz_arr_m=mic_xyz_arr,
                mic_xyz_room_m=mic_xyz_room,
                geom_jitter_mm=( (mic_xyz_arr - mic_positions_uca4(cfg["array"]["radius_m"])) * 1000.0 ),
                sim_defaults=sim_defaults,
                model=_absorption_model(rt60_s),
            )

            for k, az_deg in enumerate(az_set):
                pose_seed = derive_seed(room_seed, k)
                rng_pose = np.random.default_rng(pose_seed)

                # New: sample (d, h_s) from a guaranteed-feasible interval for this azimuth
                try:
                    d, h_s, (d_min, d_max) = sample_distance_and_height_feasible(
                        rng_pose, dims, pose, cfg, az_deg=float(az_deg)
                    )
                except RuntimeError:
                    # Very rare: if no feasible height found, skip this az and continue (or resample another az)
                    continue

                # Place source and, if near-zenith, try to push d toward d_max
                clear = cfg["sim"]["clearance_m"]
                max_push = 8
                while True:
                    try:
                        src_xyz = place_source_from_array_center(float(az_deg), float(d), float(h_s), pose)
                    except Exception:
                        # increase d a little within bounds
                        d = min(d_max, 1.1 * d)
                        if d >= d_max * 0.999:
                            # fall back: resample feasible (d,h_s)
                            try:
                                d, h_s, (d_min, d_max) = sample_distance_and_height_feasible(
                                    rng_pose, dims, pose, cfg, az_deg=float(az_deg)
                                )
                                continue
                            except RuntimeError:
                                break
                        continue

                    if not _validate_inside_room(src_xyz, dims, clear):
                        d = min(d_max, 1.05 * d)
                        if d >= d_max * 0.999:
                            break
                        continue

                    u_arr, az_lab, el_lab, is_nz = labels_from_geometry(
                        src_xyz, pose.center_m, pose, cfg["sim"]["near_zenith_min_upar"]
                    )
                    if is_nz and max_push > 0:
                        max_push -= 1
                        d = min(d_max, 1.15 * d)  # push outward to increase planar component
                        if d >= d_max * 0.999:
                            # give up on this height; try a new feasible (d,h_s)
                            try:
                                d, h_s, (d_min, d_max) = sample_distance_and_height_feasible(
                                    rng_pose, dims, pose, cfg, az_deg=float(az_deg)
                                )
                                max_push = 8
                                continue
                            except RuntimeError:
                                break
                        continue
                    break  # success or non-recoverable

                # If we failed to place, skip this azimuth
                try:
                    _ = src_xyz  # noqa
                except UnboundLocalError:
                    continue

                rir_len_s = clamp_rir_len(rt60_s, cfg["sim"]["rir_len_max_s"])

                room_spec = (L, W, H, rt60_s, _absorption_model(rt60_s))
                rirs = generate_rir_pra(
                    room_spec=room_spec,
                    mic_xyz_room=mic_xyz_room,
                    src_xyz_room=src_xyz,
                    fs_hz=args.fs,
                    rir_len_s=rir_len_s,
                    cfg=cfg,
                )

                rt60_ms = int(round(1000.0 * rt60_s))
                rir_ms = int(round(1000.0 * rir_len_s))
                d_cm = 100.0 * float(d)
                pose_dir = make_pose_dir(out_root, args.split, room_id, cfg["array"]["id"], rt60_ms, k)
                base_fn = rir_filename(room_id, rt60_ms, rdeg, az_lab, el_lab, d_cm, args.fs, rir_ms, pose_seed, mic_index=0)
                save_wavs(pose_dir, base_fn, rirs, args.fs)

                room_meta_rel = os.path.relpath(room_meta_path, pose_dir)
                write_pose_meta(
                    pose_dir=pose_dir,
                    room_meta_rel=room_meta_rel,
                    unit_vec_array=u_arr,
                    az_deg=float(az_lab),
                    el_deg=float(el_lab),
                    distance_m=float(d),
                    src_xyz_room_m=src_xyz,
                    method="ISM",
                    rir_len_s=rir_len_s,
                    seed=pose_seed,
                )

            room_counter += 1
            pbar.update(1)
    pbar.close()

if __name__ == "__main__":
    main()
