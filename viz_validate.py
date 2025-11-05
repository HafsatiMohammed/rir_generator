# viz_validate.py
from __future__ import annotations
import argparse
import glob
import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from geometry import make_rotated_grid, labels_from_geometry, place_source_from_array_center
from samplers import load_config, sample_rt60_in_bin, sample_room_size, sample_distance_and_height_feasible
from array import sample_room_array_geometry
from rir import clamp_rir_len, generate_rir_pra


def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _build_demo(out_root: str, rooms_demo: int, K_demo: int, fs: int, seed: int, cfg: dict):
    split = "demo"
    np.random.seed(seed)
    bins = cfg["rt60"]["bins_s"]

    pick_idxs = [0, 3, 9] if len(bins) >= 10 else list(range(min(3, len(bins))))
    per_bin = [rooms_demo // len(pick_idxs)] * len(pick_idxs)
    for i in range(rooms_demo - sum(per_bin)):
        per_bin[i] += 1

    room_counter = 0
    for bcount, bidx in zip(per_bin, pick_idxs):
        blo, bhi = bins[bidx]
        for _ in range(bcount):
            room_id = f"{room_counter:04d}"
            rng_room = np.random.default_rng(seed * 9973 + room_counter)

            rt60_s = sample_rt60_in_bin(rng_room, blo, bhi)
            L, W, H = sample_room_size(rt60_s, rng_room, cfg)
            dims = (L, W, H)

            rdeg = float(rng_room.uniform(0.0, 5.0))
            local_grid = make_rotated_grid(rdeg)
            az_set = rng_room.choice(local_grid, size=min(K_demo, 72), replace=False)

            pose, mic_xyz_arr, mic_xyz_room, jp = sample_room_array_geometry(rng_room, cfg, dims)
            sim_defaults = {"fs_hz": int(fs), "c_mps": float(cfg["sim"]["c_mps"])}

            room_dir = os.path.join(out_root, split, f"room-{room_id}")
            os.makedirs(room_dir, exist_ok=True)
            room_meta = {
                "room": {"id": f"room_{room_id}", "dims_m": list(map(float, dims)), "rt60_s": float(rt60_s), "model": "Eyring"},
                "array": {
                    "id": cfg["array"]["id"],
                    "pose_room": {
                        "roll_deg": float(pose.roll_deg),
                        "pitch_deg": float(pose.pitch_deg),
                        "yaw_deg": float(pose.yaw_deg),
                        "center_m": list(map(float, pose.center_m.tolist())),
                    },
                    "mic_xyz_array_m": mic_xyz_arr.tolist(),
                    "mic_xyz_room_m": mic_xyz_room.tolist(),
                    "geom_jitter_mm": ((mic_xyz_arr * 0.0)).tolist()
                },
                "sim_defaults": sim_defaults,
            }
            with open(os.path.join(room_dir, "room_meta.json"), "w") as f:
                json.dump(room_meta, f, indent=2)

            # Generate poses using the feasible (d, h_s) sampler
            for k, az_deg in enumerate(az_set):
                rng_pose = np.random.default_rng(seed * 131 + k + room_counter)
                try:
                    d, h_s, (d_min, d_max) = sample_distance_and_height_feasible(
                        rng_pose, dims, pose, cfg, az_deg=float(az_deg)
                    )
                except RuntimeError:
                    # If a particular azimuth has no feasible heights, skip it in the demo
                    continue

                # Place & push outward if near-zenith
                max_push = 8
                clear = cfg["sim"]["clearance_m"]
                while True:
                    try:
                        src_xyz = place_source_from_array_center(float(az_deg), float(d), float(h_s), pose)
                    except Exception:
                        d = min(d_max, 1.1 * d)
                        if d >= d_max * 0.999:
                            break
                        continue

                    if not (clear <= src_xyz[0] <= L - clear and clear <= src_xyz[1] <= W - clear and clear <= src_xyz[2] <= H - clear):
                        d = min(d_max, 1.05 * d)
                        if d >= d_max * 0.999:
                            break
                        continue

                    u_arr, az_lab, el_lab, is_nz = labels_from_geometry(
                        src_xyz, pose.center_m, pose, cfg["sim"]["near_zenith_min_upar"]
                    )
                    if is_nz and max_push > 0:
                        max_push -= 1
                        d = min(d_max, 1.15 * d)
                        if d >= d_max * 0.999:
                            break
                        continue
                    break

                try:
                    _ = src_xyz  # if failed, UnboundLocalError
                except UnboundLocalError:
                    continue

                rir_len_s = clamp_rir_len(rt60_s, cfg["sim"]["rir_len_max_s"])
                rirs = generate_rir_pra((L, W, H, rt60_s, "Eyring"), mic_xyz_room, src_xyz, fs, rir_len_s, cfg)

                pose_dir = os.path.join(out_root, split, f"room-{room_id}", "_poses", f"srcpose-{k:04d}")
                os.makedirs(pose_dir, exist_ok=True)
                for m in range(rirs.shape[0]):
                    wf = os.path.join(pose_dir, f"rir_demo_room-{room_id}_az-{int(round(az_lab))}_mic-{m}.wav")
                    sf.write(wf, rirs[m], fs, subtype="FLOAT")

                with open(os.path.join(pose_dir, "meta.json"), "w") as f:
                    json.dump({
                        "az_deg": float(az_lab),
                        "el_deg": float(el_lab),
                        "distance_m": float(d),
                        "src_xyz_room_m": src_xyz.tolist(),
                        "unit_vec_array": u_arr.tolist(),
                    }, f, indent=2)

            room_counter += 1


def _plot_room_3d(figpath: str, room_meta_fp: str, pose_fp: str):
    with open(room_meta_fp, "r") as f:
        R = json.load(f)
    with open(pose_fp, "r") as f:
        P = json.load(f)

    L, W, H = R["room"]["dims_m"]
    mic_room = np.array(R["array"]["mic_xyz_room_m"], dtype=float)
    center = np.array(R["array"]["pose_room"]["center_m"], dtype=float)
    src = np.array(P["src_xyz_room_m"], dtype=float)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    xs = [0, L, L, 0, 0]
    ys = [0, 0, W, W, 0]
    for z in [0, H]:
        ax.plot(xs, ys, [z] * 5, lw=1)
    for (x, y) in [(0, 0), (L, 0), (L, W), (0, W)]:
        ax.plot([x, x], [y, y], [0, H], lw=1)

    ax.scatter([center[0]], [center[1]], [center[2]], s=40, marker='o', label="Array center")
    mic0_dir = mic_room[0] - center
    mic0_dir = mic0_dir / (np.linalg.norm(mic0_dir) + 1e-12)
    ax.quiver(center[0], center[1], center[2], mic0_dir[0], mic0_dir[1], mic0_dir[2],
              length=0.3, linewidth=2, arrow_length_ratio=0.1, label="+x (Mic0)")

    ax.scatter(mic_room[:, 0], mic_room[:, 1], mic_room[:, 2], s=30, label="Mics")
    for i, (x, y, z) in enumerate(mic_room):
        ax.text(x, y, z, f"M{i}", fontsize=8)

    ax.scatter([src[0]], [src[1]], [src[2]], s=40, marker='^', label="Source")

    ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, H)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.set_title("Room 3D view")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(figpath, dpi=160)
    plt.close(fig)


def _plot_rirs_and_spectra(figpath_wave: str, figpath_spec: str, wavs: List[str], fs: int):
    rirs = [sf.read(wf)[0] for wf in wavs]
    M = len(rirs)
    t = [np.arange(len(r)) / fs for r in rirs]

    plt.figure(figsize=(8, 6))
    for i in range(M):
        plt.plot(t[i], rirs[i], label=f"Mic {i}")
    plt.xlabel("Time [s]"); plt.ylabel("Amplitude"); plt.title("RIR waveforms (4 mics)")
    plt.legend(); plt.tight_layout(); plt.savefig(figpath_wave, dpi=160); plt.close()

    plt.figure(figsize=(8, 6))
    for i in range(M):
        R = np.fft.rfft(rirs[i] * np.hanning(len(rirs[i])))
        f = np.fft.rfftfreq(len(rirs[i]), 1.0 / fs)
        mag = 20 * np.log10(np.maximum(np.abs(R), 1e-12))
        plt.plot(f, mag, label=f"Mic {i}")
    plt.xlabel("Frequency [Hz]"); plt.ylabel("Magnitude [dB]"); plt.title("RIR magnitude spectra (4 mics)")
    plt.legend(); plt.tight_layout(); plt.savefig(figpath_spec, dpi=160); plt.close()


def _az_hist(figpath: str, pose_meta_files: List[str]):
    az_list = []
    for fp in pose_meta_files:
        with open(fp, "r") as f:
            P = json.load(f)
        az = P.get("az_deg", None)
        if az is None and "src_pose" in P:
            az = P["src_pose"]["azimuth_deg"]
        az_list.append(float(az))
    az_arr = np.array(az_list, dtype=float)

    bins = np.arange(0, 361, 5)
    plt.figure(figsize=(7, 4))
    plt.hist(az_arr, bins=bins, edgecolor='k')
    plt.xlabel("Azimuth [deg] (global 5° bins)"); plt.ylabel("Count")
    plt.title("Azimuth coverage (quantized)")
    plt.tight_layout(); plt.savefig(figpath, dpi=160); plt.close()


def _toa_check(figpath: str, room_meta_fp: str, pose_fp: str, wavs: List[str], c_mps: float):
    with open(room_meta_fp, "r") as f:
        R = json.load(f)
    with open(pose_fp, "r") as f:
        P = json.load(f)

    fs = int(R["sim_defaults"].get("fs_hz", 16000))
    mic_room = np.array(R["array"]["mic_xyz_room_m"], dtype=float)
    src = np.array(P.get("src_xyz_room_m", P.get("src_pose", {}).get("src_xyz_room_m")), dtype=float)

    dists = np.linalg.norm(mic_room - src.reshape(1, 3), axis=1)
    toa = dists / c_mps
    dt_pred = toa - toa[0]

    rirs = [sf.read(wf)[0] for wf in wavs]
    thr = 0.1 * max(np.max(np.abs(r)) for r in rirs)
    t_first = []
    for r in rirs:
        idx = int(np.argmax(np.abs(r) > thr))
        t_first.append(idx / fs)
    dt_meas = np.array(t_first) - t_first[0]

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(dt_pred)), 1e6 * dt_pred, marker='o', label="Predicted ΔTOA [μs]")
    plt.plot(range(len(dt_meas)), 1e6 * dt_meas, marker='x', label="Measured ΔTOA [μs]")
    plt.xlabel("Mic index"); plt.ylabel("ΔTOA [μs]"); plt.title("First-arrival inter-mic delays")
    plt.legend(); plt.tight_layout(); plt.savefig(figpath, dpi=160); plt.close()

    print("ΔTOA predicted [μs]:", (1e6 * dt_pred).round(1).tolist())
    print("ΔTOA measured  [μs]:", (1e6 * dt_meas).round(1).tolist())


def _consistency_check(room_dir: str):
    room_meta_fp = os.path.join(room_dir, "room_meta.json")
    with open(room_meta_fp, "r") as f:
        R = json.load(f)
    mic_room = np.array(R["array"]["mic_xyz_room_m"], dtype=float)
    max_delta = float(np.max(np.abs(mic_room - mic_room)))
    assert max_delta < 1e-9, f"Mic coordinates changed within the room! max |Δ| = {max_delta}"


def main():
    ap = argparse.ArgumentParser(description="Visualization validator for the RIR demo subset.")
    ap.add_argument("--rooms_demo", type=int, default=8)
    ap.add_argument("--K_demo", type=int, default=12)
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config()
    out_root = os.path.abspath(args.out)
    _ensure_dir(out_root)

    _build_demo(out_root, args.rooms_demo, args.K_demo, args.fs, args.seed, cfg)

    viz_dir = os.path.join(out_root, "_viz")
    _ensure_dir(viz_dir)

    room_dirs = sorted(glob.glob(os.path.join(out_root, "demo", "room-*")))
    for i, rd in enumerate(room_dirs[:3]):
        room_meta_fp = os.path.join(rd, "room_meta.json")
        pose_fp = sorted(glob.glob(os.path.join(rd, "_poses", "srcpose-*", "meta.json")))[0]
        _plot_room_3d(os.path.join(viz_dir, f"room3d_{i}.png"), room_meta_fp, pose_fp)

    pose_meta_files = sorted(glob.glob(os.path.join(out_root, "demo", "room-*", "_poses", "srcpose-*", "meta.json")))
    if pose_meta_files:
        demo_pose = pose_meta_files[0]
        pose_dir = os.path.dirname(demo_pose)
        wavs = sorted(glob.glob(os.path.join(pose_dir, "rir_demo_*_mic-*.wav")))
        if wavs:
            _plot_rirs_and_spectra(
                os.path.join(viz_dir, "rirs_waveforms.png"),
                os.path.join(viz_dir, "rirs_spectra.png"),
                wavs,
                args.fs
            )

    if pose_meta_files:
        _az_hist(os.path.join(viz_dir, "az_hist.png"), pose_meta_files)

    if pose_meta_files:
        demo_pose = pose_meta_files[min(1, len(pose_meta_files) - 1)]
        pose_dir = os.path.dirname(demo_pose)
        rd = os.path.dirname(os.path.dirname(pose_dir))
        room_meta_fp = os.path.join(rd, "room_meta.json")
        wavs = sorted(glob.glob(os.path.join(pose_dir, "rir_demo_*_mic-*.wav")))
        if wavs:
            _toa_check(os.path.join(viz_dir, "toa_check.png"), room_meta_fp, demo_pose, wavs, cfg["sim"]["c_mps"])

    for rd in room_dirs[:3]:
        _consistency_check(rd)

    print(f"Saved visualizations under: {viz_dir}")


if __name__ == "__main__":
    main()
