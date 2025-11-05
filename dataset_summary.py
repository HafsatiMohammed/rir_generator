#!/usr/bin/env python3
from __future__ import annotations
import argparse
import glob
import json
import os
from collections import defaultdict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
from tqdm import tqdm

# --------- Configurable aesthetics (pure Matplotlib, no seaborn needed) ---------
PLT_FACE = "#0f1117"       # dark background (slate)
AX_FACE  = "#141820"
FG       = "#e6eaf2"       # text/lines
GRID     = "#2a2f3a"
PALETTE  = ["#7aa2f7", "#9ece6a", "#f7768e", "#e0af68"]  # blue, green, pink, amber
HEATMAP_CMAP = "viridis"
ALPHA_FILL   = 0.35

GLOBAL_AZ_BINS = np.arange(0, 361, 5)  # 72 bins
RT60_EDGES     = np.linspace(0.1, 1.2, 12 + 1)
DIST_EDGES     = np.linspace(0.5, 5.0, 23)
HEIGHT_EDGES   = np.linspace(1.0, 2.0, 21)

SMALL = {"L": (3.0, 5.0), "W": (3.0, 5.0), "H": (2.4, 3.0)}
MEDIUM = {"L": (6.0, 10.0), "W": (5.0, 8.0), "H": (3.0, 4.0)}
LARGE = {"L": (12.0, 20.0), "W": (10.0, 15.0), "H": (4.0, 6.0)}


@dataclass(frozen=True)
class RoomInfo:
    split: str
    room_dir: str
    room_id: str
    dims: Tuple[float, float, float]  # (L, W, H)
    rt60_s: float
    array_center: Tuple[float, float, float]
    mic_xyz_room: np.ndarray  # (4, 3)
    array_pitch_deg: Optional[float] = None
    array_roll_deg: Optional[float] = None
    array_yaw_deg: Optional[float] = None
    array_id: Optional[str] = None


@dataclass(frozen=True)
class PoseInfo:
    split: str
    room_id: str
    az_deg: float
    el_deg: Optional[float]
    distance_m: Optional[float]
    src_xyz_room: Optional[Tuple[float, float, float]]


# ----------------------------- utilities -----------------------------

def _ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def _within(x: float, lo: float, hi: float, eps: float = 1e-6) -> bool:
    return (x >= lo - eps) and (x <= hi + eps)


def classify_room_size(dims: Tuple[float, float, float]) -> str:
    L, W, H = dims
    def _is(bucket):
        return _within(L, *bucket["L"]) and _within(W, *bucket["W"]) and _within(H, *bucket["H"])
    if _is(SMALL):  return "Small"
    if _is(MEDIUM): return "Medium"
    if _is(LARGE):  return "Large"
    return "Other"


def _find_room_dirs(root: str, split: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, split, "room-*")))


def _load_room_meta(room_dir: str) -> Optional[RoomInfo]:
    fp = os.path.join(room_dir, "room_meta.json")
    if not os.path.isfile(fp):
        return None
    with open(fp, "r") as f:
        J = json.load(f)

    dims = tuple(float(x) for x in J["room"]["dims_m"])
    center = tuple(float(x) for x in J["array"]["pose_room"]["center_m"])
    mics = np.array(J["array"]["mic_xyz_room_m"], dtype=float)
    pitch = float(J["array"]["pose_room"].get("pitch_deg", 0.0))
    roll = float(J["array"]["pose_room"].get("roll_deg", 0.0))
    yaw = float(J["array"]["pose_room"].get("yaw_deg", 0.0))
    rid = os.path.basename(room_dir).replace("room-", "")
    arr_id = J["array"].get("id", "array")
    rt60_s = float(J["room"]["rt60_s"])
    split = os.path.basename(os.path.dirname(room_dir))

    return RoomInfo(
        split=split, room_dir=room_dir, room_id=rid,
        dims=dims, rt60_s=rt60_s,
        array_center=center, mic_xyz_room=mics,
        array_pitch_deg=pitch, array_roll_deg=roll, array_yaw_deg=yaw, array_id=arr_id
    )


def _list_pose_meta_files(room_dir: str) -> List[str]:
    prod = glob.glob(os.path.join(room_dir, "array-*", "rt60-*", "srcpose-*", "meta.json"))
    if prod: return sorted(prod)
    demo = glob.glob(os.path.join(room_dir, "_poses", "srcpose-*", "meta.json"))
    return sorted(demo)


def _load_pose_meta(split: str, room_id: str, pose_fp: str) -> Optional[PoseInfo]:
    with open(pose_fp, "r") as f:
        P = json.load(f)

    if "az_deg" in P:
        az = float(P["az_deg"])
        el = float(P.get("el_deg", np.nan)) if "el_deg" in P else None
        d  = float(P.get("distance_m", np.nan)) if "distance_m" in P else None
        src = tuple(P.get("src_xyz_room_m", [np.nan, np.nan, np.nan]))
    else:
        sp = P.get("src_pose", {})
        az = float(sp.get("azimuth_deg", np.nan))
        el = float(sp.get("elevation_deg", np.nan)) if "elevation_deg" in sp else None
        d  = float(sp.get("distance_m", np.nan)) if "distance_m" in sp else None
        src = tuple(sp.get("src_xyz_room_m", [np.nan, np.nan, np.nan]))
    return PoseInfo(split=split, room_id=room_id, az_deg=az, el_deg=el, distance_m=d, src_xyz_room=src)


# ----------------------------- styling -----------------------------

def _set_fig_style():
    plt.rcParams.update({
        "axes.facecolor": AX_FACE,
        "figure.facecolor": PLT_FACE,
        "savefig.facecolor": PLT_FACE,
        "axes.edgecolor": FG, "axes.labelcolor": FG, "text.color": FG,
        "xtick.color": FG, "ytick.color": FG,
        "grid.color": GRID, "grid.linestyle": "--", "grid.linewidth": 0.6,
        "axes.grid": True,
        "legend.frameon": False,
        "font.size": 11,
        "figure.dpi": 120,
    })


def _overlay_hist(ax, data, edges, label, color, density=True, fill=True, alpha=ALPHA_FILL):
    if len(data) == 0:
        return
    hist, bins = np.histogram(data, bins=edges, density=density)
    centers = 0.5 * (bins[:-1] + bins[1:])
    ax.plot(centers, hist, color=color, label=label, lw=2)
    if fill:
        ax.fill_between(centers, 0, hist, color=color, alpha=alpha)


def _stacked_hist(ax, by_split: Dict[str, List[float]], edges, split_order, colors, density=False):
    # compute counts per split per bin
    counts = []
    for s in split_order:
        d = by_split.get(s, [])
        h, _ = np.histogram(d, bins=edges, density=False)
        counts.append(h.astype(float))
    counts = np.vstack(counts) if counts else np.zeros((0, len(edges)-1))
    if counts.size == 0:
        return
    totals = counts.sum(axis=0)
    if density and totals.sum() > 0:
        # normalize columns to probability mass
        with np.errstate(divide="ignore", invalid="ignore"):
            counts = counts / (totals.sum())
    left = np.zeros(len(edges)-1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    for i, s in enumerate(split_order):
        ax.bar(centers, counts[i], bottom=left, width=width, color=colors[i], label=s, align="center", edgecolor="none")
        left += counts[i]


def _styled_save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ----------------------------- plotting packs -----------------------------

def plot_overview_rt60_az(out_dir: str, rooms_by_split, poses_by_split):
    _set_fig_style()
    splits = [s for s in ("train", "val", "test", "demo") if s in rooms_by_split]

    # Collect RT60 per split
    rt60_by_split = {s: [r.rt60_s for r in rooms_by_split[s]] for s in splits}
    # Collect azimuths per split
    az_by_split   = {s: [p.az_deg for p in poses_by_split[s] if np.isfinite(p.az_deg)] for s in splits}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle("Dataset Overview", color=FG, fontsize=14, y=0.98)

    # RT60: stacked per split (counts)
    _stacked_hist(ax1, rt60_by_split, RT60_EDGES, splits, PALETTE[:len(splits)], density=False)
    ax1.set_xlabel("RT60 [s]"); ax1.set_ylabel("Count")
    ax1.set_title("RT60 distribution (stacked by split)")
    ax1.xaxis.set_major_locator(MaxNLocator(12))
    ax1.legend(loc="upper right")

    # Azimuth: overlay (density)
    for i, s in enumerate(splits):
        _overlay_hist(ax2, az_by_split[s], GLOBAL_AZ_BINS, label=s, color=PALETTE[i], density=True, fill=True)
    ax2.set_xlabel("Azimuth [deg] (5° bins)"); ax2.set_ylabel("Probability density")
    ax2.set_title("Azimuth distribution (overlay)")
    ax2.set_xlim(0, 360)
    ax2.legend(loc="upper right")

    _styled_save(fig, os.path.join(out_dir, "overview_rt60_az.png"))


def plot_sources_distributions(out_dir: str, poses_by_split):
    _set_fig_style()
    splits = [s for s in ("train", "val", "test", "demo") if s in poses_by_split]

    # Prepare data
    d_by_split = {s: [p.distance_m for p in poses_by_split[s] if (p.distance_m is not None and np.isfinite(p.distance_m) and p.distance_m > 0)]
                  for s in splits}
    h_by_split = {s: [p.src_xyz_room[2] for p in poses_by_split[s] if (p.src_xyz_room is not None and all(np.isfinite(p.src_xyz_room)))]
                  for s in splits}
    az_all     = [p.az_deg for s in splits for p in poses_by_split[s] if np.isfinite(p.az_deg)]
    dist_all   = [p.distance_m for s in splits for p in poses_by_split[s] if (p.distance_m is not None and np.isfinite(p.distance_m))]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6))
    ((ax_d, ax_h), (ax_hm, ax_leg)) = axes
    fig.suptitle("Source Distributions", color=FG, fontsize=14, y=0.98)

    # Distances overlay
    for i, s in enumerate(splits):
        _overlay_hist(ax_d, d_by_split[s], DIST_EDGES, label=s, color=PALETTE[i], density=True, fill=True)
    ax_d.set_xlabel("Source distance [m]"); ax_d.set_ylabel("Probability density")
    ax_d.set_title("Distance (overlay)")
    ax_d.legend(loc="upper right")

    # Heights overlay
    for i, s in enumerate(splits):
        _overlay_hist(ax_h, h_by_split[s], HEIGHT_EDGES, label=s, color=PALETTE[i], density=True, fill=True)
    ax_h.set_xlabel("Source height [m]"); ax_h.set_ylabel("Probability density")
    ax_h.set_title("Height (overlay)")
    ax_h.legend(loc="upper right")

    # Azimuth × Distance heatmap (all splits)
    if az_all and dist_all:
        H, xedges, yedges = np.histogram2d(az_all, dist_all, bins=[GLOBAL_AZ_BINS, DIST_EDGES])
        im = ax_hm.imshow(
            H.T, origin="lower", aspect="auto",
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
            cmap=HEATMAP_CMAP
        )
        ax_hm.set_xlabel("Azimuth [deg]"); ax_hm.set_ylabel("Distance [m]")
        ax_hm.set_title("Azimuth × Distance density (all splits)")
        cbar = fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)
        cbar.set_label("Count")

    # Decorative legend panel (empty axes)
    ax_leg.axis("off")
    msg = "Notes:\n• Overlays show per-split densities (area ≈ 1).\n• Heatmap aggregates all splits."
    ax_leg.text(0.02, 0.98, msg, va="top", ha="left", color=FG, family="monospace")

    _styled_save(fig, os.path.join(out_dir, "sources_dists.png"))


def plot_room_sizes_and_pose(out_dir: str, rooms_by_split):
    _set_fig_style()
    splits = [s for s in ("train", "val", "test", "demo") if s in rooms_by_split]

    # Sizes per split
    sizes_by_split = {}
    for s in splits:
        sizes = [classify_room_size(r.dims) for r in rooms_by_split[s]]
        sizes_by_split[s] = Counter(sizes)

    # Center heights & pitch per split (for overlays)
    center_by_split = {s: [r.array_center[2] for r in rooms_by_split[s]] for s in splits}
    pitch_by_split  = {s: [r.array_pitch_deg for r in rooms_by_split[s] if r.array_pitch_deg is not None] for s in splits}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.6))
    (ax_sizes, ax_note), (ax_center, ax_pitch) = axes
    fig.suptitle("Room Classes & Array Pose", color=FG, fontsize=14, y=0.98)

    # Stacked bars for sizes (one bar per split)
    classes = ["Small", "Medium", "Large", "Other"]
    x = np.arange(len(splits))
    bottoms = np.zeros(len(splits))
    for i, cls in enumerate(classes):
        vals = np.array([sizes_by_split[s].get(cls, 0) for s in splits], dtype=float)
        total = vals.sum()
        ax_sizes.bar(x, vals, bottom=bottoms, color=PALETTE[i % len(PALETTE)], label=cls)
        bottoms += vals
    ax_sizes.set_xticks(x); ax_sizes.set_xticklabels(splits)
    ax_sizes.set_ylabel("Room count"); ax_sizes.set_title("Room size classes (stacked)")
    ax_sizes.legend(loc="upper right")

    # Overlay: center height distributions
    for i, s in enumerate(splits):
        _overlay_hist(ax_center, center_by_split[s], np.linspace(1.0, 1.8, 17), label=s, color=PALETTE[i], density=True, fill=True)
    ax_center.set_xlabel("Array center height [m]"); ax_center.set_ylabel("Probability density")
    ax_center.set_title("Array center height (overlay)")
    ax_center.legend(loc="upper right")

    # Overlay: pitch distributions
    for i, s in enumerate(splits):
        _overlay_hist(ax_pitch, pitch_by_split[s], np.linspace(0, 50, 26), label=s, color=PALETTE[i], density=True, fill=True)
    ax_pitch.set_xlabel("Array pitch [deg]"); ax_pitch.set_ylabel("Probability density")
    ax_pitch.set_title("Array pitch (overlay)")
    ax_pitch.legend(loc="upper right")

    # Little notes
    ax_note.axis("off")
    ax_note.text(0.02, 0.98,
                 "• Stacked bars: actual counts per split.\n"
                 "• Overlays: normalized densities.", va="top", ha="left",
                 color=FG, family="monospace")

    _styled_save(fig, os.path.join(out_dir, "room_sizes_and_pose.png"))


def plot_mic_showcase_topdown(out_dir: str, rooms_by_split):
    _set_fig_style()
    splits = [s for s in ("train", "val", "test", "demo") if s in rooms_by_split]

    for s in splits:
        Rs = rooms_by_split[s]
        if not Rs:
            continue
        show_n = min(8, len(Rs))
        idxs = np.linspace(0, len(Rs) - 1, num=show_n, dtype=int)
        cols = 4
        rows = int(np.ceil(show_n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.4*cols, 3.2*rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, i in zip(axes, idxs):
            r = Rs[i]
            L, W, _ = r.dims
            # room boundary
            ax.fill_between([0, L], 0, W, color="#1a1f29", alpha=0.7, step="pre")
            ax.plot([0, L, L, 0, 0], [0, 0, W, W, 0], color=FG, lw=1.0)
            # array center and mics
            ax.scatter([r.array_center[0]], [r.array_center[1]], c=PALETTE[0], s=26, label="Center")
            ax.scatter(r.mic_xyz_room[:,0], r.mic_xyz_room[:,1], c=PALETTE[1], s=24, label="Mics")
            # mic labels, subtly
            for mi, (x, y, _) in enumerate(r.mic_xyz_room):
                ax.text(x, y, f"M{mi}", color=FG, fontsize=8, ha="left", va="bottom")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(-0.05, L+0.05); ax.set_ylim(-0.05, W+0.05)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{s}: room-{r.room_id}", fontsize=10, color=FG, pad=2)
        # turn off unused axes
        for ax in axes[len(idxs):]:
            ax.axis("off")
        fig.suptitle(f"Mic placement — top-down ({s})", color=FG, fontsize=14, y=0.98)
        _styled_save(fig, os.path.join(out_dir, f"mic_showcase_topdown_{s}.png"))


# ----------------------------- main summarizer -----------------------------

def summarize_dataset(root: str, out_dir: str):
    splits_present = [s for s in ("train", "val", "test", "demo") if os.path.isdir(os.path.join(root, s))]
    if not splits_present:
        raise SystemExit(f"No split folders found under {root} (expected train/val/test or demo).")
    _ensure_dir(out_dir)

    rooms_by_split: Dict[str, List[RoomInfo]] = defaultdict(list)
    poses_by_split: Dict[str, List[PoseInfo]] = defaultdict(list)

    # Scan once
    for split in splits_present:
        room_dirs = _find_room_dirs(root, split)
        for rd in tqdm(room_dirs, desc=f"[scan] {split} rooms", unit="room"):
            R = _load_room_meta(rd)
            if R is None:
                continue
            rooms_by_split[split].append(R)
            for pm in _list_pose_meta_files(rd):
                P = _load_pose_meta(split, R.room_id, pm)
                if P is not None:
                    poses_by_split[split].append(P)

    # Plots
    plot_overview_rt60_az(out_dir, rooms_by_split, poses_by_split)
    plot_sources_distributions(out_dir, poses_by_split)
    plot_room_sizes_and_pose(out_dir, rooms_by_split)
    plot_mic_showcase_topdown(out_dir, rooms_by_split)

    print(f"[summary] Saved figures in: {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Beautiful grouped summaries for an RIR dataset (train/val/test).")
    ap.add_argument("--root", required=True, help="Path to rir_bank root (contains train/val/test or demo)")
    ap.add_argument("--out", required=True, help="Directory to save summary figures")
    args = ap.parse_args()
    summarize_dataset(args.root, args.out)


if __name__ == "__main__":
    main()
