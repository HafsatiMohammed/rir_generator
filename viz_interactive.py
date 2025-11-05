from __future__ import annotations
import argparse
import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback_context
import soundfile as sf

# import your project modules
from geometry import (
    Pose,
    place_source_from_array_center,
    labels_from_geometry,
)
# Optional: if you saved make_rotated_grid etc, not needed here


# ---------- helpers ----------
def _load_rooms(root: str, split: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, split, "room-*")))

def _load_room_meta(room_dir: str) -> Dict:
    with open(os.path.join(room_dir, "room_meta.json"), "r") as f:
        return json.load(f)

def _room_pose_from_meta(meta: Dict) -> Pose:
    p = meta["array"]["pose_room"]
    return Pose(
        roll_deg=float(p["roll_deg"]),
        pitch_deg=float(p["pitch_deg"]),
        yaw_deg=float(p["yaw_deg"]),
        center_m=np.array(p["center_m"], dtype=float),
    )

def _list_pose_meta(room_dir: str) -> List[str]:
    # Works for demo (“_poses/.../meta.json”) and production (nested structure)
    demo = sorted(glob.glob(os.path.join(room_dir, "_poses", "srcpose-*", "meta.json")))
    if demo:
        return demo
    # production layout
    return sorted(glob.glob(os.path.join(room_dir, "array-*", "rt60-*", "srcpose-*", "meta.json")))

def _read_rirs_for_pose(pose_meta_fp: str) -> Tuple[List[np.ndarray], int]:
    # Demo naming: rir_demo_room-XXXX_az-YYY_mic-M.wav
    pose_dir = os.path.dirname(pose_meta_fp)
    wavs = sorted(glob.glob(os.path.join(pose_dir, "rir_*_mic-*.wav")))
    rirs = []
    fs = None
    for wf in wavs:
        sig, fs_ = sf.read(wf)
        if fs is None:
            fs = fs_
        rirs.append(sig.astype(np.float64))
    return rirs, fs or 16000

def _room_box_traces(L, W, H, color="rgba(150,150,150,0.4)") -> List[go.Scatter3d]:
    # Draw edges of a rectangular prism
    xs = [0, L, L, 0, 0]
    ys = [0, 0, W, W, 0]
    traces = []
    for z in [0,H]:
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=[z]*5, mode="lines", line=dict(color=color, width=2), showlegend=False
        ))
    for (x,y) in [(0,0),(L,0),(L,W),(0,W)]:
        traces.append(go.Scatter3d(
            x=[x,x], y=[y,y], z=[0,H], mode="lines", line=dict(color=color, width=2), showlegend=False
        ))
    return traces

def _arrow_trace(p0: np.ndarray, p1: np.ndarray, name: str, color: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
        mode="lines+markers",
        line=dict(color=color, width=6),
        marker=dict(size=3, color=color),
        name=name
    )

def _plot_room(meta: Dict, pose_meta_fp: str, whatif: Dict|None=None) -> go.Figure:
    L, W, H = meta["room"]["dims_m"]
    center = np.array(meta["array"]["pose_room"]["center_m"], dtype=float)
    mics = np.array(meta["array"]["mic_xyz_room_m"], dtype=float)

    # Get saved source from meta (demo or production schema)
    with open(pose_meta_fp, "r") as f:
        PM = json.load(f)
    src_saved = np.array(
        PM.get("src_xyz_room_m", PM.get("src_pose", {}).get("src_xyz_room_m")),
        dtype=float
    )

    fig = go.Figure()
    # room
    for tr in _room_box_traces(L, W, H):
        fig.add_trace(tr)
    # array center
    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode="markers", marker=dict(size=6, color="blue"),
        name="Array center"
    ))
    # mics
    fig.add_trace(go.Scatter3d(
        x=mics[:,0], y=mics[:,1], z=mics[:,2],
        mode="markers+text",
        marker=dict(size=5, color="orange"),
        text=[f"M{i}" for i in range(len(mics))],
        textposition="top center",
        name="Mics"
    ))
    # saved source
    fig.add_trace(go.Scatter3d(
        x=[src_saved[0]], y=[src_saved[1]], z=[src_saved[2]],
        mode="markers", marker=dict(size=6, color="green"),
        name="Saved source"
    ))
    # saved direction arrow (approx from center toward source)
    fig.add_trace(_arrow_trace(center, src_saved, "Saved ray", "green"))

    # what-if overlay (if provided)
    if whatif is not None and whatif.get("ok", False):
        src_new = whatif["src"]
        fig.add_trace(go.Scatter3d(
            x=[src_new[0]], y=[src_new[1]], z=[src_new[2]],
            mode="markers", marker=dict(size=6, color="red"),
            name="What-if source"
        ))
        fig.add_trace(_arrow_trace(center, src_new, "What-if ray", "red"))

    fig.update_layout(
        scene=dict(
            xaxis_title="X [m]", yaxis_title="Y [m]", zaxis_title="Z [m]",
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

def _compute_whatif(
    az_deg: float, d_m: float, h_s: float,
    pose: Pose, dims: Tuple[float,float,float], clear: float, nz_thresh: float
) -> Dict:
    L,W,H = dims
    try:
        src = place_source_from_array_center(az_deg, d_m, h_s, pose)
    except Exception:
        return {"ok": False, "msg": "Infeasible combination (az,d,h) for this pose."}
    x,y,z = src
    inside = (clear <= x <= L-clear) and (clear <= y <= W-clear) and (clear <= z <= H-clear)
    u_arr, az_lab, el_lab, is_nz = labels_from_geometry(src, pose.center_m, pose, nz_thresh)
    # room-frame elevation (extra intuition)
    u_room = pose.R_array_to_room() @ u_arr
    el_room_deg = float(np.degrees(np.arcsin(np.clip(u_room[2], -1.0, 1.0))))
    return {
        "ok": inside,
        "src": src,
        "az_lab": float(az_lab),
        "el_arr_deg": float(el_lab),
        "el_room_deg": el_room_deg,
        "near_zenith": bool(is_nz),
        "msg": ("" if inside else "Out of bounds (clearance).")
    }

def _time_series_fig(rirs: List[np.ndarray], fs: int) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(rirs):
        t = np.arange(len(r)) / fs
        fig.add_trace(go.Scatter(x=t, y=r, name=f"Mic {i}", mode="lines"))
    fig.update_layout(
        title="RIR waveforms",
        xaxis_title="Time [s]", yaxis_title="Amplitude",
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

def _mag_spec_fig(rirs: List[np.ndarray], fs: int) -> go.Figure:
    fig = go.Figure()
    for i, r in enumerate(rirs):
        w = np.hanning(len(r))
        R = np.fft.rfft(r * w)
        f = np.fft.rfftfreq(len(r), 1.0/fs)
        mag = 20*np.log10(np.maximum(np.abs(R), 1e-12))
        fig.add_trace(go.Scatter(x=f, y=mag, name=f"Mic {i}", mode="lines"))
    fig.update_layout(
        title="RIR magnitude spectra",
        xaxis_title="Frequency [Hz]", yaxis_title="Magnitude [dB]",
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig


# ---------- Dash app ----------
def main():
    parser = argparse.ArgumentParser(description="Interactive RIR room visualizer.")
    parser.add_argument("--root", required=True, help="rir_bank root (e.g., ./rir_bank_demo)")
    parser.add_argument("--split", default="demo", help="split folder (demo/train/val/test)")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    rooms = _load_rooms(args.root, args.split)
    if not rooms:
        raise SystemExit(f"No rooms found under {args.root}/{args.split}")

    # Preload minimal meta for all rooms
    rooms_meta = {rd: _load_room_meta(rd) for rd in rooms}
    room_to_poses = {rd: _list_pose_meta(rd) for rd in rooms}

    app = Dash(__name__)
    app.title = "RIR Interactive Visualizer"

    def room_options():
        return [{"label": os.path.basename(rd), "value": rd} for rd in rooms]

    def pose_options(room_dir: str):
        poses = room_to_poses.get(room_dir, [])
        return [{"label": os.path.basename(os.path.dirname(p)), "value": p} for p in poses]

    first_room = rooms[0]
    first_pose = room_to_poses[first_room][0]

    # UI
    app.layout = html.Div([
        html.H3("RIR Interactive Visualizer"),
        html.Div([
            html.Div([
                html.Label("Room"),
                dcc.Dropdown(id="room-dropdown", options=room_options(), value=first_room, clearable=False),
            ], style={"width":"30%", "display":"inline-block", "verticalAlign":"top", "marginRight":"1rem"}),

            html.Div([
                html.Label("Pose"),
                dcc.Dropdown(id="pose-dropdown", options=pose_options(first_room), value=first_pose, clearable=False),
            ], style={"width":"30%", "display":"inline-block", "verticalAlign":"top", "marginRight":"1rem"}),

            html.Div([
                html.Label("What-if (source in array frame)"),
                html.Div([
                    html.Label("Azimuth [deg]"),
                    dcc.Slider(id="az-slider", min=0, max=360, step=1, value=45,
                               marks=None, tooltip={"placement":"bottom", "always_visible":True}),
                    html.Label("Distance d [m]"),
                    dcc.Slider(id="d-slider", min=0.5, max=5.0, step=0.01, value=2.0,
                               marks=None, tooltip={"placement":"bottom", "always_visible":True}),
                    html.Label("Height h_s [m] (room frame)"),
                    dcc.Slider(id="h-slider", min=1.2, max=1.8, step=0.005, value=1.6,
                               marks=None, tooltip={"placement":"bottom", "always_visible":True}),
                ]),
                html.Div(id="whatif-readout", style={"marginTop":"0.5rem", "fontFamily":"monospace"}),
            ], style={"width":"36%", "display":"inline-block", "verticalAlign":"top"}),
        ], style={"marginBottom":"1rem"}),

        dcc.Graph(id="room-graph", style={"height":"58vh"}),

        html.Div([
            dcc.Graph(id="rir-waves", style={"width":"49%", "display":"inline-block", "height":"34vh"}),
            dcc.Graph(id="rir-spec",  style={"width":"49%", "display":"inline-block", "height":"34vh"}),
        ])
    ], style={"margin":"0.8rem"})

    # Update pose dropdown when room changes
    @app.callback(
        Output("pose-dropdown", "options"),
        Output("pose-dropdown", "value"),
        Input("room-dropdown", "value"),
    )
    def _on_room_change(room_dir):
        opts = pose_options(room_dir)
        val = opts[0]["value"] if opts else None
        return opts, val

    # Main update: 3D graph + what-if readout + RIR plots
    @app.callback(
        Output("room-graph", "figure"),
        Output("whatif-readout", "children"),
        Output("rir-waves", "figure"),
        Output("rir-spec", "figure"),
        Input("room-dropdown", "value"),
        Input("pose-dropdown", "value"),
        Input("az-slider", "value"),
        Input("d-slider", "value"),
        Input("h-slider", "value"),
    )
    def _update(room_dir, pose_meta_fp, az_deg, d_m, h_s):
        meta = rooms_meta[room_dir]
        pose = _room_pose_from_meta(meta)
        dims = tuple(meta["room"]["dims_m"])
        clear = float(0.30)  # from config; hardcode for speed here
        nz_thresh = 0.10

        # compute what-if placement/labels
        wi = _compute_whatif(float(az_deg), float(d_m), float(h_s), pose, dims, clear, nz_thresh)
        readout = []
        if wi["ok"]:
            readout = [
                f"OK  |  az_lab: {wi['az_lab']:.2f}°  |  el_arr: {wi['el_arr_deg']:.2f}°  |  el_room: {wi['el_room_deg']:.2f}°",
                f"near_zenith: {wi['near_zenith']}  |  src_xyz_room: " +
                f"[{wi['src'][0]:.3f}, {wi['src'][1]:.3f}, {wi['src'][2]:.3f}]"
            ]
        else:
            readout = [f"NOT PLACED: {wi['msg']}"]

        fig3d = _plot_room(meta, pose_meta_fp, wi if wi["ok"] else None)

        # RIR plots for the selected SAVED pose
        rirs, fs = _read_rirs_for_pose(pose_meta_fp)
        waves = _time_series_fig(rirs, fs) if rirs else go.Figure()
        spec  = _mag_spec_fig(rirs, fs) if rirs else go.Figure()
        return fig3d, html.Pre("\n".join(readout)), waves, spec

    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
