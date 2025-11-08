# RIR Generator

A comprehensive Room Impulse Response (RIR) generator for Direction of Arrival (DoA) estimation tasks. This tool generates synthetic RIR datasets using the Image Source Method (ISM) via PyRoomAcoustics, simulating realistic acoustic environments with configurable room geometries, RT60 values, microphone array configurations, and source positions.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Format](#output-format)
- [Visualization Tools](#visualization-tools)
- [Dataset Information](#dataset-information)
- [Technical Details](#technical-details)

## Overview

This RIR generator is designed for training and evaluating DoA estimation models. It creates synthetic acoustic environments with:

- **Variable room geometries**: Small, medium, and large rooms with configurable dimensions
- **Realistic RT60 values**: Configurable reverberation times (0.1s to 1.2s)
- **4-microphone UCA array**: Uniform Circular Array with 27.7mm radius
- **Geometric jitter**: Realistic microphone position variations
- **Multiple source positions**: Configurable azimuth/elevation distributions
- **Comprehensive metadata**: Full geometric and acoustic parameters for each RIR

## Features

### Core Capabilities

- **Room Simulation**: Generate rooms with configurable sizes (small: 3-5m, medium: 6-10m, large: 12-20m)
- **RT60 Control**: Precise RT60 binning with configurable distributions across train/val/test splits
- **Array Geometry**: 4-microphone UCA with configurable pose (roll, pitch, yaw) and position
- **Source Placement**: Intelligent source positioning with feasibility checks
- **Geometric Jitter**: Realistic microphone position variations (mixture model with clean-room option)
- **Image Source Method**: High-quality RIR generation using PyRoomAcoustics
- **Metadata Export**: Complete JSON metadata for each room and source pose

### Advanced Features

- **Feasible Sampling**: Guaranteed valid source positions for given azimuth/elevation constraints
- **Near-Zenith Handling**: Special handling for sources near the array's zenith
- **Distance Prior Options**: Uniform, log-uniform, or power-law distance distributions
- **Automatic ISM Order**: Adaptive reflection order based on room size and RT60
- **Deterministic Generation**: Seed-based reproducibility

## Installation

### Requirements

**Core Dependencies:**
- Python 3.8+
- NumPy
- PyRoomAcoustics
- PyYAML
- soundfile
- tqdm

**Optional Dependencies:**
- matplotlib (for visualization tools)
- plotly and dash (for interactive visualization)
- PyTorch and torchaudio (for `mix_batcher.py` audio mixing utilities)

### Install Dependencies

```bash
pip install numpy pyroomacoustics pyyaml soundfile tqdm matplotlib plotly dash
```

Or create a `requirements.txt`:

```txt
# Core dependencies
numpy>=1.20.0
pyroomacoustics>=0.7.0
pyyaml>=5.4.0
soundfile>=0.10.0
tqdm>=4.60.0

# Visualization (optional)
matplotlib>=3.3.0
plotly>=5.0.0
dash>=2.0.0

# Audio mixing utilities (optional)
torch>=1.10.0
torchaudio>=0.10.0
```

Then install:

```bash
pip install -r requirements.txt
```

## Project Structure

```
rir_generator/
├── cli_generate.py          # Main CLI for RIR generation
├── config.yaml              # Configuration file
├── rir.py                   # RIR generation using PyRoomAcoustics
├── array.py                 # Microphone array geometry and jitter
├── geometry.py              # Geometric transformations and source placement
├── samplers.py              # Sampling functions for rooms, RT60, distances
├── io_meta.py               # Metadata I/O functions
├── dataset_summary.py       # Dataset analysis and visualization
├── viz_interactive.py       # Interactive visualization tool
├── viz_validate.py          # Validation visualization
├── mix_batcher.py           # Audio mixing utilities
├── main_test_tdoa.py        # TDOA testing utilities
└── rir_bank/                # Generated RIR dataset
    ├── train/
    ├── val/
    └── test/
```

## Configuration

The `config.yaml` file controls all aspects of RIR generation:

### Simulation Parameters

```yaml
sim:
  c_mps: 343.0                    # Speed of sound (m/s)
  default_fs_hz: 16000            # Default sampling rate
  clearance_m: 0.30               # Minimum distance to walls
  rir_len_max_s: 1.20             # Maximum RIR length
  max_order_default: auto         # ISM reflection order (auto or integer)
  max_order_cap: 20               # Maximum reflection order
```

### Array Configuration

```yaml
array:
  id: uca4_r27p7mm                # Array identifier
  radius_m: 0.0277                 # UCA radius (27.7mm)
  center_near_room_center: true    # Keep array near room center
  center_radius_m: 0.5             # Radius around room center
  center_height_range_m: [1.2, 1.5] # Array height range
  yaw_deg_range: [0.0, 360.0]     # Yaw angle range
  pitch_tilt_deg_mixture:          # Pitch tilt distribution
    - { range: [35.0, 40.0], p: 0.25 }
    - { range: [40, 45.0], p: 0.50 }
    - { range: [45.0, 50.0], p: 0.25 }
  clean_room_probability: 0.08    # Probability of no jitter
```

### Room Sizes

```yaml
rooms:
  small:
    L_range: [3.0, 5.0]
    W_range: [3.0, 5.0]
    H_range: [2.4, 3.0]
  medium:
    L_range: [6.0, 10.0]
    W_range: [5.0, 8.0]
    H_range: [3.0, 4.0]
  large:
    L_range: [12.0, 20.0]
    W_range: [10.0, 15.0]
    H_range: [4.0, 6.0]
```

### RT60 Distribution

```yaml
rt60:
  bins_s:                         # RT60 bins (seconds)
    - [0.1, 0.2]
    - [0.2, 0.3]
    # ... more bins
  counts:
    train: [75, 120, 180, ...]    # Room counts per bin
    val: [10, 16, 24, ...]
    test: [15, 24, 36, ...]
```

### Source Configuration

```yaml
distances:
  range_m: [0.5, 5.0]              # Distance range
  prior: log_uniform_feasible      # Distance prior type
  radial_frac_of_min_wh: 0.80     # Max radial distance

heights:
  speech_mu_m: 1.70                # Mean speech height
  speech_sigma_m: 0.07             # Height standard deviation
  clip_range_m: [1.50, 2.00]      # Height clipping range
```

## Usage

### Basic Generation

Generate RIRs for a specific split:

```bash
python cli_generate.py \
    --split train \
    --rooms 1500 \
    --fs 16000 \
    --K 36 \
    --seed 42 \
    --out ./rir_bank
```

**Arguments:**
- `--split`: Dataset split (`train`, `val`, or `test`)
- `--rooms`: Number of rooms to generate
- `--fs`: Sampling rate in Hz (default: 16000)
- `--K`: Number of source poses per room (default: 36)
- `--seed`: Random seed for reproducibility
- `--out`: Output directory path

### Generation Process

1. **Room Sampling**: For each room, samples:
   - RT60 value from configured bins
   - Room size (L×W×H) based on RT60 pairing
   - Array pose (position, roll, pitch, yaw)
   - Microphone jitter parameters

2. **Source Placement**: For each source pose:
   - Samples azimuth from rotated grid (72 bins, 5° resolution)
   - Samples feasible distance and height
   - Validates source position (clearance, near-zenith handling)
   - Generates RIR using ISM

3. **Output**: Saves:
   - RIR WAV files (one per microphone)
   - Room metadata JSON
   - Source pose metadata JSON

### Example: Generate All Splits

```bash
# Training set
python cli_generate.py --split train --rooms 1500 --seed 42 --out ./rir_bank

# Validation set
python cli_generate.py --split val --rooms 200 --seed 43 --out ./rir_bank

# Test set
python cli_generate.py --split test --rooms 300 --seed 44 --out ./rir_bank
```

## Output Format

### Directory Structure

```
rir_bank/
├── train/
│   └── room-0000/
│       ├── room_meta.json
│       └── array-uca4_r27p7mm/
│           └── rt60-150/
│               └── srcpose-0000/
│                   ├── meta.json
│                   ├── rir_room-0000_rt60-150_rot-0_az-0_el-15_d-120_fs-16000_len-195_seed-12345_mic-0.wav
│                   ├── rir_room-0000_rt60-150_rot-0_az-0_el-15_d-120_fs-16000_len-195_seed-12345_mic-1.wav
│                   ├── rir_room-0000_rt60-150_rot-0_az-0_el-15_d-120_fs-16000_len-195_seed-12345_mic-2.wav
│                   └── rir_room-0000_rt60-150_rot-0_az-0_el-15_d-120_fs-16000_len-195_seed-12345_mic-3.wav
└── ...
```

### Room Metadata (`room_meta.json`)

```json
{
  "room": {
    "id": "room_0000",
    "dims_m": [4.5, 4.2, 2.8],
    "rt60_s": 0.15,
    "model": "Eyring"
  },
  "array": {
    "id": "uca4_r27p7mm",
    "pose_room": {
      "roll_deg": 0.5,
      "pitch_deg": 42.3,
      "yaw_deg": 45.0,
      "center_m": [2.25, 2.10, 1.35]
    },
    "mic_xyz_array_m": [[...], [...], [...], [...]],
    "mic_xyz_room_m": [[...], [...], [...], [...]],
    "geom_jitter_mm": [[...], [...], [...], [...]],
    "clean_room": false
  },
  "sim_defaults": {
    "fs_hz": 16000,
    "c_mps": 343.0
  }
}
```

### Source Pose Metadata (`meta.json`)

```json
{
  "room_meta": "../../room_meta.json",
  "src_pose": {
    "direction_unit_xyz_array": [0.707, 0.707, 0.0],
    "azimuth_deg": 45.0,
    "elevation_deg": 0.0,
    "distance_m": 1.2,
    "src_xyz_room_m": [3.1, 2.9, 1.7]
  },
  "sim": {
    "method": "ISM",
    "rir_len_s": 0.195,
    "seed": 12345
  }
}
```

## Visualization Tools

### Dataset Summary

Generate comprehensive statistics and visualizations:

```bash
python dataset_summary.py --rir_bank ./rir_bank --out ./_summary_viz
```

Creates visualizations for:
- RT60 distribution
- Room size distribution
- Azimuth/elevation histograms
- Distance distributions
- Array pose distributions

### Interactive Visualization

Launch an interactive web-based visualization:

```bash
python viz_interactive.py --rir_bank ./rir_bank
```

Features:
- 3D room visualization
- RIR waveform display
- Spectrum analysis
- Source/array positioning
- Interactive navigation

### Validation Visualization

Validate generated RIRs:

```bash
python viz_validate.py --rir_bank ./rir_bank --room_id 0000
```

### Audio Mixing Utilities

The `mix_batcher.py` module provides utilities for:
- Convolving speech signals with RIRs
- Mixing multiple sources
- Adding background noise
- Batch processing for training data generation

Requires PyTorch and torchaudio (optional).

## Dataset Information

### Default Configuration

- **Total Rooms**: 2000 (1500 train, 200 val, 300 test)
- **Source Poses per Room**: 36 (configurable via `--K`)
- **RT60 Range**: 0.1s to 1.2s (11 bins)
- **Azimuth Resolution**: 5° (72 bins, 0-360°)
- **Sampling Rate**: 16 kHz (configurable)
- **Array**: 4-microphone UCA, 27.7mm radius

### RT60 Distribution

The RT60 values are distributed across bins with different counts per split:
- **Low RT60 (0.1-0.3s)**: More common in small rooms
- **Medium RT60 (0.3-0.8s)**: Balanced across room sizes
- **High RT60 (0.8-1.2s)**: More common in large rooms

### Room Size Pairing

Room sizes are paired with RT60 values:
- **0.1-0.3s**: 70% small, 25% medium, 5% large
- **0.3-0.8s**: 35% small, 50% medium, 15% large
- **0.8-1.2s**: 10% small, 45% medium, 45% large

## Technical Details

### Image Source Method (ISM)

RIRs are generated using PyRoomAcoustics' ISM implementation:
- **Reflection Order**: Automatically computed based on room size and RT60
- **Absorption Model**: Eyring model (default) or Sabine
- **Air Absorption**: Enabled
- **RIR Length**: `min(1.3 × RT60, 1.2s)`

### Coordinate Systems

- **Room Frame**: Origin at one corner, Z-up
- **Array Frame**: Origin at array center, Z-up (normal to array plane)
- **Transformations**: ZYX extrinsic Euler angles (yaw-pitch-roll)

### Source Placement Algorithm

1. Sample azimuth from rotated grid (72 bins, 5° resolution)
2. Sample height from Gaussian distribution (μ=1.70m, σ=0.07m)
3. Compute feasible distance interval `[d_min, d_max]` for given azimuth/height
4. Sample distance from configured prior (log-uniform by default)
5. Validate source position (clearance, near-zenith handling)
6. Generate RIR using ISM

### Geometric Jitter

Microphone positions include realistic manufacturing variations:
- **Mixture Model**: 3-component Gaussian mixture
  - 80%: σ=2mm, clip=5mm
  - 15%: σ=4mm, clip=8mm
  - 5%: σ=6-8mm (uniform), clip=2×σ
- **Clean Room**: 8% probability of zero jitter

### Feasibility Guarantees

The generator ensures:
- All sources are within room boundaries (with clearance)
- All sources have valid geometric solutions
- Near-zenith sources are handled appropriately
- Distance/height combinations are physically realizable

## Contributing

This project is designed for DoA estimation research. Contributions are welcome for:
- Additional array geometries
- Alternative absorption models
- Enhanced visualization tools
- Performance optimizations


## Citation

If you use this RIR generator in your research, please cite:

```bibtex
@software{rir_generator,
  title = {RIR Generator for DoA Estimation},
  author = {[Your Name/Organization]},
  year = {2024},
  url = {[Repository URL]}
}
```

## Acknowledgments

- PyRoomAcoustics for ISM implementation
- LibriSpeech for speech dataset
- FreeSound and SoundBible for noise datasets
