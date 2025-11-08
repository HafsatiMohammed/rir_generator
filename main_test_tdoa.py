# interactive_sources_min.py
from __future__ import annotations
import math
from typing import Tuple
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from array import mic_positions_uca4, DEFAULT_UCA4_RADIUS_M
from geometry import Pose, DEG2RAD

# ---------------------------- constants ----------------------------
C_MPS = 343.0
ROOM_DIMS = (6.0, 5.0, 3.0)   # L, W, H [m]
CLEARANCE_M = 0.5             # >= 0.5 m from walls
RT60S = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]  # 8 labels for export

FIGSIZE = (12, 7)
DPI = 110

# Defaults
ROLL0, PITCH0, YAW0 = 45.0, 45.0, 20.0
S1_R0, S1_AZ0, S1_EL0 = 1.0, 40.0, +20.0
S2_R0, S2_AZ0, S2_EL0 = 2.0, 40.0, -20.0

# ---------------------------- helpers ----------------------------
def unit_from_azel_array(az_deg: float, el_deg: float) -> np.ndarray:
    az = az_deg * DEG2RAD
    el = el_deg * DEG2RAD
    ce = math.cos(el)
    return np.array([ce * math.cos(az), ce * math.sin(az), math.sin(el)], float)

def inside_with_clearance(xyz: np.ndarray, room_dims: Tuple[float, float, float],
                          clearance_m: float) -> bool:
    L, W, H = room_dims
    x, y, z = map(float, xyz)
    return (clearance_m <= x <= L - clearance_m and
            clearance_m <= y <= W - clearance_m and
            clearance_m <= z <= H - clearance_m)

def draw_room_box(ax, L, W, H):
    edges = []
    edges += [((0, 0, 0), (L, 0, 0)), ((L, 0, 0), (L, W, 0)),
              ((L, W, 0), (0, W, 0)), ((0, W, 0), (0, 0, 0))]
    edges += [((0, 0, H), (L, 0, H)), ((L, 0, H), (L, W, H)),
              ((L, W, H), (0, W, H)), ((0, W, H), (0, 0, H))]
    edges += [((0, 0, 0), (0, 0, H)), ((L, 0, 0), (L, 0, H)),
              ((L, W, 0), (L, W, H)), ((0, W, 0), (0, W, H))]
    ax.add_collection3d(Line3DCollection(edges, linewidths=1.0, alpha=0.35))

def mic_xyz_room(pose: Pose, radius_m: float) -> np.ndarray:
    base = mic_positions_uca4(radius_m)  # (4,3) array frame
    R = pose.R_array_to_room()
    return (R @ base.T).T + pose.center_m.reshape(1, 3)

def compute_src_xyz(pose: Pose, r_m: float, az_deg: float, el_deg: float) -> np.ndarray:
    R = pose.R_array_to_room()
    ctr = pose.center_m
    u = unit_from_azel_array(az_deg, el_deg)
    return ctr + R @ (r_m * u)

def compute_tdoa_rel0(mic_xyz_room_: np.ndarray, src_xyz_room: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(mic_xyz_room_ - src_xyz_room[None, :], axis=1)
    return (d - d[0]) / C_MPS

# ---------------------------- app ----------------------------
def interactive_app():
    matplotlib.use('TkAgg')  # switch to 'Qt5Agg' if you prefer Qt

    L, W, H = ROOM_DIMS

    # Fixed off-center array center
    arr_center = np.array([0.35 * L, 0.60 * W, min(H - CLEARANCE_M - 0.7, 1.2)], float)

    # Figure & axes
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)

    # Main plots
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)
    status_txt = fig.text(0.02, 0.14, "", fontsize=9)

    # ---------- TOP: Src1 (left) ----------
    ax_s1r  = plt.axes([0.08, 0.92, 0.35, 0.02], facecolor='lightgoldenrodyellow')
    ax_s1az = plt.axes([0.08, 0.89, 0.35, 0.02], facecolor='lightgoldenrodyellow')
    ax_s1el = plt.axes([0.08, 0.86, 0.35, 0.02], facecolor='lightgoldenrodyellow')
    s1_r  = Slider(ax_s1r,  'Src1 r [m]', 0.2, 6.0, valinit=S1_R0)
    s1_az = Slider(ax_s1az, 'Src1 Az°',   0.0, 359.9, valinit=S1_AZ0)
    s1_el = Slider(ax_s1el, 'Src1 El°',  -89.0, 89.0, valinit=S1_EL0)

    # ---------- TOP: Src2 (right) ----------
    ax_s2r  = plt.axes([0.57, 0.92, 0.35, 0.02], facecolor='lightgoldenrodyellow')
    ax_s2az = plt.axes([0.57, 0.89, 0.35, 0.02], facecolor='lightgoldenrodyellow')
    ax_s2el = plt.axes([0.57, 0.86, 0.35, 0.02], facecolor='lightgoldenrodyellow')
    s2_r  = Slider(ax_s2r,  'Src2 r [m]', 0.2, 6.0, valinit=S2_R0)
    s2_az = Slider(ax_s2az, 'Src2 Az°',   0.0, 359.9, valinit=S2_AZ0)
    s2_el = Slider(ax_s2el, 'Src2 El°',  -89.0, 89.0, valinit=S2_EL0)

    # ---------- BOTTOM: roll, pitch, yaw ----------
    ax_roll  = plt.axes([0.08, 0.08, 0.24, 0.03], facecolor='lightgoldenrodyellow')
    ax_pitch = plt.axes([0.36, 0.08, 0.24, 0.03], facecolor='lightgoldenrodyellow')
    ax_yaw   = plt.axes([0.64, 0.08, 0.20, 0.03], facecolor='lightgoldenrodyellow')
    s_roll  = Slider(ax_roll,  'Roll°',  0.0, 89.0,   valinit=ROLL0)
    s_pitch = Slider(ax_pitch, 'Pitch°', 0.0, 89.0,   valinit=PITCH0)
    s_yaw   = Slider(ax_yaw,   'Yaw°',   0.0, 359.9,  valinit=YAW0)

    # ---------- BOTTOM-RIGHT: lock + buttons ----------
    ax_lock  = plt.axes([0.86, 0.085, 0.11, 0.05], facecolor='lightgoldenrodyellow')
    ax_rc    = plt.axes([0.86, 0.045, 0.11, 0.03])
    ax_batch = plt.axes([0.86, 0.010, 0.11, 0.03])
    lock_chk = CheckButtons(ax_lock, ['Lock Array Pose'], [True])  # default locked (stable)
    b_recenter = Button(ax_rc, 'Recenter Array')
    b_batch    = Button(ax_batch, 'Batch Save (8 RT60)')

    # Convenience
    array_sliders = [s_roll, s_pitch, s_yaw]
    def set_array_sliders_active(active: bool):
        for s in array_sliders:
            try: s.set_active(active)
            except Exception: pass

    # start locked
    set_array_sliders_active(False)

    # ---------------------------- drawing ----------------------------
    def update_plot(_=None):
        # Build pose (orientation may be locked)
        pose = Pose(
            roll_deg=s_roll.val,
            pitch_deg=s_pitch.val,
            yaw_deg=s_yaw.val,
            center_m=arr_center.copy(),
        )
        mic_xyz = mic_xyz_room(pose, DEFAULT_UCA4_RADIUS_M)
        s1_xyz = compute_src_xyz(pose, s1_r.val, s1_az.val, s1_el.val)
        s2_xyz = compute_src_xyz(pose, s2_r.val, s2_az.val, s2_el.val)

        # Checks (≥ 0.5 m clearance)
        flags = []
        if not all(inside_with_clearance(m, ROOM_DIMS, CLEARANCE_M) for m in mic_xyz):
            flags.append("Mic(s) violate clearance")
        if not inside_with_clearance(s1_xyz, ROOM_DIMS, CLEARANCE_M):
            flags.append("Src1 violates clearance")
        if not inside_with_clearance(s2_xyz, ROOM_DIMS, CLEARANCE_M):
            flags.append("Src2 violates clearance")

        # --- 3D ---
        ax3d.cla()
        draw_room_box(ax3d, *ROOM_DIMS)
        ax3d.set_xlim(0, L); ax3d.set_ylim(0, W); ax3d.set_zlim(0, H)
        ax3d.set_xlabel("x [m]"); ax3d.set_ylabel("y [m]"); ax3d.set_zlabel("z [m]")
        ax3d.view_init(elev=18, azim=35)
        # mics & cross
        ax3d.scatter(mic_xyz[:, 0], mic_xyz[:, 1], mic_xyz[:, 2], s=45, label="Mics")
        for i, j in [(0, 2), (1, 3)]:
            ax3d.plot([mic_xyz[i, 0], mic_xyz[j, 0]],
                      [mic_xyz[i, 1], mic_xyz[j, 1]],
                      [mic_xyz[i, 2], mic_xyz[j, 2]], linewidth=1.2)
        # center
        ax3d.scatter([arr_center[0]], [arr_center[1]], [arr_center[2]], s=30, marker='x', label="Array center")
        # sources
        ax3d.scatter([s1_xyz[0]], [s1_xyz[1]], [s1_xyz[2]], s=70, marker='o', label="Src1")
        ax3d.scatter([s2_xyz[0]], [s2_xyz[1]], [s2_xyz[2]], s=70, marker='s', label="Src2")
        ax3d.legend(loc="upper left", fontsize=8)

        # --- TDOA ---
        ax2d.cla()
        x_idx = np.arange(4)
        t1 = compute_tdoa_rel0(mic_xyz, s1_xyz) * 1e3
        t2 = compute_tdoa_rel0(mic_xyz, s2_xyz) * 1e3
        ax2d.plot(x_idx, t1, marker='o', label="Src1")
        ax2d.plot(x_idx, t2, marker='s', label="Src2")
        ax2d.set_xticks(x_idx)
        ax2d.set_xlabel("Mic channel index")
        ax2d.set_ylabel("TDOA rel. Mic0 [ms]")
        ax2d.grid(True, alpha=0.3)
        ax2d.legend()

        # status
        msg = (f"Room [6.0,5.0,3.0] m | Clearance 0.5 m | "
               f"Roll {s_roll.val:.1f}°, Pitch {s_pitch.val:.1f}°, Yaw {s_yaw.val:.1f}° | "
               f"S1 r={s1_r.val:.2f}m az={s1_az.val:.1f}° el={s1_el.val:.1f}° | "
               f"S2 r={s2_r.val:.2f}m az={s2_az.val:.1f}° el={s2_el.val:.1f}°")
        if flags:
            msg += " | WARN: " + " ; ".join(flags)
        status_txt.set_text(msg)

        fig.canvas.draw_idle()

    def on_lock(_label):
        # toggle active state of orientation sliders
        active = not lock_chk.get_status()[0]
        set_array_sliders_active(active)

    def on_recenter(_):
        # put array off-center & valid (center is fixed in this UI)
        nonlocal arr_center
        arr_center = np.array([0.35 * L, 0.60 * W, min(H - CLEARANCE_M - 0.7, 1.2)], float)
        update_plot()

    def plot_case(rt60_s: float,
                  mic_xyz_room_: np.ndarray,
                  array_center_: np.ndarray,
                  s1_xyz_: np.ndarray,
                  s2_xyz_: np.ndarray,
                  room_dims_: Tuple[float, float, float],
                  savepath: str):
        Lx, Wx, Hx = room_dims_
        t1 = compute_tdoa_rel0(mic_xyz_room_, s1_xyz_) * 1e3
        t2 = compute_tdoa_rel0(mic_xyz_room_, s2_xyz_) * 1e3
        x_idx = np.arange(mic_xyz_room_.shape[0])

        f = plt.figure(figsize=(12, 6), dpi=110)
        f.suptitle(
            f"RT60={rt60_s:.2f}s | roll={s_roll.val:.1f}°, pitch={s_pitch.val:.1f}°, yaw={s_yaw.val:.1f}° | "
            f"S1(r={s1_r.val:.2f}, az={s1_az.val:.1f}°, el={s1_el.val:.1f}°) | "
            f"S2(r={s2_r.val:.2f}, az={s2_az.val:.1f}°, el={s2_el.val:.1f}°)",
            y=0.98
        )

        axA = f.add_subplot(1, 2, 1, projection='3d')
        draw_room_box(axA, Lx, Wx, Hx)
        axA.scatter(mic_xyz_room_[:, 0], mic_xyz_room_[:, 1], mic_xyz_room_[:, 2], s=40, label="Mics")
        for i, j in [(0, 2), (1, 3)]:
            axA.plot([mic_xyz_room_[i, 0], mic_xyz_room_[j, 0]],
                     [mic_xyz_room_[i, 1], mic_xyz_room_[j, 1]],
                     [mic_xyz_room_[i, 2], mic_xyz_room_[j, 2]], linewidth=1.2)
        axA.scatter([array_center_[0]], [array_center_[1]], [array_center_[2]],
                    s=25, marker='x', label="Array center")
        axA.scatter([s1_xyz_[0]], [s1_xyz_[1]], [s1_xyz_[2]], s=60, marker='o', label="Src1")
        axA.scatter([s2_xyz_[0]], [s2_xyz_[1]], [s2_xyz_[2]], s=60, marker='s', label="Src2")
        axA.set_xlim(0, Lx); axA.set_ylim(0, Wx); axA.set_zlim(0, Hx)
        axA.set_xlabel("x [m]"); axA.set_ylabel("y [m]"); axA.set_zlabel("z [m]")
        axA.view_init(elev=18, azim=35)
        axA.legend(loc="upper left", fontsize=8)

        axB = f.add_subplot(1, 2, 2)
        axB.plot(x_idx, t1, marker='o', label="Src1")
        axB.plot(x_idx, t2, marker='s', label="Src2")
        axB.set_xticks(x_idx)
        axB.set_xlabel("Mic channel index")
        axB.set_ylabel("TDOA rel. Mic0 [ms]")
        axB.grid(True, alpha=0.3)
        axB.legend()

        f.tight_layout(rect=[0, 0, 1, 0.95])
        f.savefig(savepath, bbox_inches="tight")
        plt.close(f)

    def on_batch(_):
        # Build pose & geometry from current UI
        pose = Pose(roll_deg=s_roll.val, pitch_deg=s_pitch.val, yaw_deg=s_yaw.val, center_m=arr_center.copy())
        mic_xyz = mic_xyz_room(pose, DEFAULT_UCA4_RADIUS_M)
        s1_xyz = compute_src_xyz(pose, s1_r.val, s1_az.val, s1_el.val)
        s2_xyz = compute_src_xyz(pose, s2_r.val, s2_az.val, s2_el.val)

        # Guards before saving
        err = []
        if not all(inside_with_clearance(m, ROOM_DIMS, CLEARANCE_M) for m in mic_xyz): err.append("mic(s)")
        if not inside_with_clearance(s1_xyz, ROOM_DIMS, CLEARANCE_M): err.append("src1")
        if not inside_with_clearance(s2_xyz, ROOM_DIMS, CLEARANCE_M): err.append("src2")
        if err:
            print(f"Batch Save aborted: clearance violation -> {', '.join(err)}")
            return

        for i, rt in enumerate(RT60S):
            path = f"room_case_{i+1:02d}_rt60_{rt:.2f}s.png"
            plot_case(rt, mic_xyz, arr_center.copy(), s1_xyz, s2_xyz, ROOM_DIMS, path)
            print(f"Saved {path}")

    # Connect callbacks
    for s in (s1_r, s1_az, s1_el, s2_r, s2_az, s2_el, s_roll, s_pitch, s_yaw):
        s.on_changed(update_plot)
    lock_chk.on_clicked(on_lock)
    b_recenter.on_clicked(on_recenter)
    b_batch.on_clicked(on_batch)

    # Layout of main subplots (so our top/bottom UI fits nicely)
    ax3d.set_position([0.06, 0.20, 0.44, 0.62])  # [left, bottom, width, height]
    ax2d.set_position([0.52, 0.20, 0.44, 0.62])

    # First render
    update_plot()
    plt.show()

# ---------------------------- entry ----------------------------
if __name__ == "__main__":
    interactive_app()

