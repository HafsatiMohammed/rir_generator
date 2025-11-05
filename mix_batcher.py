# mix_batcher.py
from __future__ import annotations
import os, json, glob, math, random, threading, functools, collections
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
from tqdm import tqdm

try:
    import torchaudio
    HAVE_TORCHAUDIO = True
except Exception:
    HAVE_TORCHAUDIO = False
import soundfile as sf



def quantize_az_deg_72(az):
    # bins {0,5,...,355}
    #return int(np.round(az / 5.0)) % 72
    return az
#############################################
# ------------ small utilities -------------
#############################################

def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def _rms(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # x: (..., T)
    return torch.sqrt(torch.clamp((x**2).mean(dim=-1), min=eps))

def _set_rms(x: torch.Tensor, target: float, eps: float = 1e-12) -> torch.Tensor:
    cur = _rms(x, eps=eps)[(...,) + (None,)]
    return x * (target / torch.clamp(cur, min=1e-6))

def _to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1: return x
    return x.mean(axis=1)

def _resample_if_needed(wave: torch.Tensor, sr_in: int, sr_out: int) -> torch.Tensor:
    if sr_in == sr_out:
        return wave
    if HAVE_TORCHAUDIO:
        return torchaudio.functional.resample(wave, sr_in, sr_out)
    # fallback: simple linear resampler
    T_in = wave.shape[-1]
    T_out = int(round(T_in * sr_out / sr_in))
    t = torch.linspace(0, T_in - 1, T_out, device=wave.device)
    t0 = torch.clamp(t.floor().long(), 0, T_in - 1)
    t1 = torch.clamp(t0 + 1, 0, T_in - 1)
    w = t - t0.float()
    return (1 - w) * wave[..., t0] + w * wave[..., t1]

def _randint_exclusive(rng: random.Random, lo: int, hi: int) -> int:
    # returns [lo, hi) like numpy
    return lo if hi <= lo else rng.randrange(lo, hi)

#############################################
# ------------ wave cache (LRU) ------------
#############################################

class WaveCache:
    """Small LRU for decoded mono audio (float32)."""
    def __init__(self, max_items: int = 256):
        self.max_items = max_items
        self.d: collections.OrderedDict[str, Tuple[np.ndarray, int]] = collections.OrderedDict()
        self.lock = threading.Lock()

    def get(self, path: str) -> Tuple[np.ndarray, int]:
        with self.lock:
            if path in self.d:
                v = self.d.pop(path)
                self.d[path] = v
                return v
        # decode outside lock
        if HAVE_TORCHAUDIO:
            wav, sr = torchaudio.load(path)   # [C, T]
            wav = wav.mean(dim=0, keepdim=False).numpy()
        else:
            wav, sr = sf.read(path, always_2d=True)
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        with self.lock:
            self.d[path] = (wav, sr)
            if len(self.d) > self.max_items:
                self.d.popitem(last=False)
        return wav, sr

#############################################
# -------------- RIR indexing --------------
#############################################

@dataclass(frozen=True)
class RIRPose:
    room_id: str
    pose_id: str
    rt60_s: float
    az_deg: float     # array-frame az
    distance_m: Optional[float]
    src_xyz: Optional[Tuple[float,float,float]]
    wavs_4: List[str] # 4 paths (mic-0..3)
    meta_path: str

class RIRBank:
    """
    Scans a bank (e.g., ./rir_bank) for a given split. Supports both your production and demo layouts.
    """
    def __init__(self, root: str, split: str):
        self.root = root
        self.split = split
        self.poses: List[RIRPose] = []
        self.rooms: Dict[str, Dict[str, Any]] = {}  # room_id -> room_meta json
        self._scan()

    def _scan(self):
        split_dir = os.path.join(self.root, self.split)
        print(split_dir)
        print(os.path.join(split_dir, "rl-*"))
        room_dirs = sorted(glob.glob(os.path.join(split_dir, "room-*")))
        print(room_dirs)
        for rd in room_dirs:
            # room meta
            rm_fp = os.path.join(rd, "room_meta.json")
            if not os.path.isfile(rm_fp): 
                continue
            with open(rm_fp, "r") as f:
                R = json.load(f)
            room_id = R["room"]["id"]
            rt60_s  = float(R["room"]["rt60_s"])
            self.rooms[room_id] = R
            #print(self.rooms)

            # production pose metas
            prod = glob.glob(os.path.join(rd, "array-*", "rt60-*", "srcpose-*", "meta.json"))
            if prod:
                for pm in prod:
                    pose_dir = os.path.dirname(pm)
                    # four mics
                    wavs = sorted(glob.glob(os.path.join(pose_dir, "*_mic-*.wav")))
                    if len(wavs) != 4: 
                        continue
                    with open(pm, "r") as f:
                        P = json.load(f)
                    src = P.get("src_pose", P)  # support demo schema too
                    az  = float(src.get("azimuth_deg", P.get("az_deg", np.nan)))
                    dist = float(src.get("distance_m", P.get("distance_m", np.nan)))
                    sxyz = src.get("src_xyz_room_m", P.get("src_xyz_room_m", None))
                    pose_id = os.path.basename(pose_dir)
                    self.poses.append(RIRPose(room_id, pose_id, rt60_s, az, dist, tuple(sxyz) if sxyz else None, wavs, pm))
                continue

            # demo pose metas
            demo = glob.glob(os.path.join(rd, "_poses", "srcpose-*", "meta.json"))
            for pm in demo:
                pose_dir = os.path.dirname(pm)
                wavs = sorted(glob.glob(os.path.join(pose_dir, "rir_demo_*_mic-*.wav")))
                if len(wavs) != 4:
                    continue
                with open(pm, "r") as f:
                    P = json.load(f)
                az  = float(P.get("az_deg", np.nan))
                dist = float(P.get("distance_m", np.nan))
                sxyz = P.get("src_xyz_room_m", None)
                pose_id = os.path.basename(pose_dir)
                self.poses.append(RIRPose(room_id, pose_id, rt60_s, az, dist, tuple(sxyz) if sxyz else None, wavs, pm))

        if not self.poses:
            raise RuntimeError(f"No RIR poses found under {self.root}/{self.split}")

    def sample_room_and_poses(
        self,
        rng: random.Random,
        n: int,
        min_sep_deg: float = 15.0,
        enforce_radial_diversity: bool = True,
    ) -> Tuple[Dict[str, Any], List[RIRPose]]:
        """
        Pick one room, then sample n distinct poses from that room with ≥ min_sep.
        """
        # group by room
        by_room: Dict[str, List[RIRPose]] = {}
        for p in self.poses:
            by_room.setdefault(p.room_id, []).append(p)
        # pick room
        room_id = rng.choice(list(by_room.keys()))
        candidates = by_room[room_id][:]
        rng.shuffle(candidates)

        chosen: List[RIRPose] = []
        for cand in candidates:
            if len(chosen) == 0:
                chosen.append(cand)
            else:
                ok = True
                for c in chosen:
                    d = abs(c.az_deg - cand.az_deg)
                    d = min(d, 360.0 - d)
                    if d < min_sep_deg: 
                        ok = False; break
                if ok:
                    chosen.append(cand)
            if len(chosen) >= n:
                break

        if len(chosen) < n:
            # fallback: allow small sep breaches rather than failing the batch
            chosen = candidates[:n]

        # optional radial diversity: use meta distances if present
        if enforce_radial_diversity:
            try:
                dists = [p.distance_m for p in chosen if p.distance_m and p.distance_m == p.distance_m]
                if len(dists) == n:
                    rratio = max(dists) / max(1e-6, min(dists))
                    if rratio < 1.25 and len(candidates) >= n:
                        # try one reshuffle to improve diversity
                        rng.shuffle(candidates)
                        chosen = candidates[:n]
            except Exception:
                pass

        return self.rooms[room_id], chosen

    def load_rirs_4(self, pose: RIRPose) -> torch.Tensor:
        """Return [4, L_rir] float32 tensor."""
        chans = []
        for w in sorted(pose.wavs_4):  # ensures mic-0..3 order
            x, sr = sf.read(w)
            if x.ndim > 1: x = x[:, 0]  # first channel if multichannel wav (should be mono)
            chans.append(torch.from_numpy(x.astype(np.float32)))
        return torch.stack(chans, dim=0)

#############################################
# --------- FFT convolution helpers --------
#############################################

def fft_conv_4ch_mono(x: torch.Tensor, rirs_4: torch.Tensor) -> torch.Tensor:
    """
    x: [T]  mono, float32
    rirs_4: [4, Lr]
    return: [4, T] same length as x (tail truncated)
    """
    T = x.numel()
    Lr = rirs_4.shape[-1]
    n_fft = _next_pow2(T + Lr - 1)
    X = torch.fft.rfft(x, n_fft)
    Y = []
    for m in range(4):
        H = torch.fft.rfft(rirs_4[m], n_fft)
        y = torch.fft.irfft(X * H, n_fft)[:T]
        Y.append(y)
    return torch.stack(Y, dim=0)

def fractional_delay_fd(y: torch.Tensor, tau_sec: float, sr: int) -> torch.Tensor:
    """
    Small fractional delay via frequency-domain phase ramp.
    y: [4, T]
    """
    if abs(tau_sec) < 1e-7: 
        return y
    T = y.shape[-1]
    n_fft = _next_pow2(T)
    freqs = torch.fft.rfftfreq(n_fft, d=1.0/sr).to(y.device)
    phase = torch.exp(-1j * 2.0 * math.pi * freqs * tau_sec)
    Y = torch.fft.rfft(y, n_fft)
    Y = Y * phase[None, :]
    return torch.fft.irfft(Y, n_fft)[..., :T]

#############################################
# ------------- augmentations --------------
#############################################

@dataclass
class AugmentCfg:
    gain_db_sigma: float = 1.0      # per-mic gain jitter
    phase_deg_sigma: float = 3.0    # per-mic constant phase
    frac_delay_sigma_us: float = 30 # per-mic fractional delay (std dev, microseconds)
    eq_prob: float = 0.5            # apply 3-band random shelves with this prob

def apply_per_mic_aug(y: torch.Tensor, sr: int, rng: random.Random, cfg: AugmentCfg) -> torch.Tensor:
    """
    y: [4, T]
    """
    # Gain jitter
    g_db = torch.tensor([rng.gauss(0.0, cfg.gain_db_sigma) for _ in range(4)], dtype=torch.float32, device=y.device)
    g = (10.0 ** (g_db / 20.0)).view(4, 1)
    y = y * g

    # Constant phase per mic (tiny)
    ph = torch.tensor([rng.gauss(0.0, cfg.phase_deg_sigma) for _ in range(4)], dtype=torch.float32, device=y.device)
    ph_rad = ph * (math.pi / 180.0)
    n_fft = _next_pow2(y.shape[-1])
    Y = torch.fft.rfft(y, n_fft)
    Y = Y * torch.exp(1j * ph_rad)[:, None]
    y = torch.fft.irfft(Y, n_fft)[..., :y.shape[-1]]

    # Fractional delay
    taus = [rng.gauss(0.0, cfg.frac_delay_sigma_us) * 1e-6 for _ in range(4)]
    for m in range(4):
        y[m] = fractional_delay_fd(y[m:m+1], taus[m], sr)[0]

    # Simple 3-band random shelf EQ (very gentle)
    if rng.random() < cfg.eq_prob:
        # piecewise frequency gains (low/mid/high)
        T = y.shape[-1]; n_fft = _next_pow2(T)
        freqs = torch.fft.rfftfreq(n_fft, d=1.0/sr).to(y.device)
        # random gains in dB
        low_db  = rng.uniform(-1.5, 1.5)
        mid_db  = rng.uniform(-1.0, 1.0)
        high_db = rng.uniform(-1.5, 1.5)
        # piecewise linear envelope
        env = torch.ones_like(freqs)
        # 0..300 Hz -> low
        env[freqs <= 300] *= 10**(low_db/20)
        # 300..4k -> mid
        mid_mask = (freqs > 300) & (freqs < 4000)
        env[mid_mask] *= 10**(mid_db/20)
        # 4k..Nyquist -> high
        env[freqs >= 4000] *= 10**(high_db/20)
        Y = torch.fft.rfft(y, n_fft) * env[None, :]
        y = torch.fft.irfft(Y, n_fft)[..., :T]

    return y

#############################################
# -------------- data indexing -------------
#############################################

def _scan_wavs(root: str) -> List[str]:
    """
    Recursively collect audio files from a dataset root that may contain
    deep subfolders (e.g., LibriSpeech speakers/chapters, Ambiances categories).
    Only change vs your original: a slightly more robust, case-insensitive scan.
    """
    if not os.path.isdir(root):
        raise RuntimeError(f"Audio root does not exist or is not a directory: {root}")

    exts = (".wav", ".flac", ".ogg")
    paths: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # ignore hidden dirs like .git
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fn in filenames:
            if fn.startswith("."):
                continue
            if os.path.splitext(fn)[1].lower() in exts:
                paths.append(os.path.join(dirpath, fn))

    if not paths:
        raise RuntimeError(f"No audio found under {root} (looked for {exts})")
    return sorted(paths)

class AudioIndex:
    def __init__(self, root: str, cache: WaveCache):
        self.root = root
        self.paths = _scan_wavs(root)
        self.cache = cache

    def sample_path(self, rng: random.Random) -> str:
        return rng.choice(self.paths)

    def get_segment(self, path: str, sr_target: int, T: int, rng: random.Random) -> torch.Tensor:
        wav, sr = self.cache.get(path)
        # pick random mono segment of length T samples at sr_target
        x = torch.from_numpy(_to_mono(wav)).float()
        if sr != sr_target:
            x = _resample_if_needed(x.unsqueeze(0), sr, sr_target).squeeze(0)
        if x.numel() < T:
            # loop
            rep = (T + x.numel() - 1) // x.numel()
            x = x.repeat(rep)[:T]
        else:
            start = _randint_exclusive(rng, 0, x.numel() - T + 1)
            x = x[start:start+T]
        return x

#############################################
# ---------- balanced epoch planner --------
#############################################

class StrictCountsPlanner:
    """
    Plans the #speakers per sample for the epoch to match exact ratios.
    Also plans the SNR bin per sample (not strictly enforced, but you can make it strict).
    """
    def __init__(self, epoch_size: int, rng: random.Random):
        self.epoch_size = epoch_size
        self.rng = rng

    def build_plan(self) -> Dict[str, List[Any]]:
        N = self.epoch_size
        counts_ratio = {1: 0.50, 2: 0.35, 3: 0.15}
        counts = {k: int(round(v * N)) for k, v in counts_ratio.items()}
        # fix rounding drift
        diff = N - sum(counts.values())
        if diff != 0:
            counts[1] += diff

        # SNR bins: not strict by default (can be strict by making a similar planner)
        snr_bins = [("clean", 0.20, (20.0, 30.0)),
                    ("medium",0.50, (5.0, 20.0)),
                    ("hard",  0.30, (-5.0, 5.0))]
        plan = []
        for k, c in counts.items():
            plan.extend([k]*c)
        self.rng.shuffle(plan)

        # snr bin per item
        snr_plan = []
        for _ in range(N):
            r = self.rng.random()
            acc = 0.0
            for name, p, rng_db in snr_bins:
                acc += p
                if r <= acc:
                    snr_plan.append((name, rng_db))
                    break

        return {"n_speech": plan, "snr_bins": snr_plan}

#############################################
# ----------- on-the-fly dataset -----------
#############################################

@dataclass
class MixConfig:
    sr: int = 16000
    dur_s: float = 4.0
    target_rms: float = 0.05  # pre-gain for each dry speech
    min_sep_deg: float = 15.0
    radial_diversity: bool = True

    ambience_K_poses: int = 6       # how many RIR poses to sum for diffuse bed
    ambience_prob_by_snr = {        # enable bed more often at harder SNRs
        "clean": 0.5,
        "medium": 0.7,
        "hard": 0.9,
    }
    local_events_poisson_by_snr = { # expected # of local events
        "clean": 0.4,
        "medium": 1.0,
        "hard": 1.6,
    }
    local_event_len_s_range = (0.2, 1.5)  # crop from local file
    clip_guard_db: float = 0.5            # mild limiter if peak > 0dBFS

class OnTheFlyMixtureDataset(torch.utils.data.Dataset):
    """
    Yields: mixture [4, T], meta dict (rich)
    """
    def __init__(
        self,
        rir_root: str,
        split: str,
        speech_root: str,
        local_noises_root: str,
        ambiences_root: str,
        epoch_size: int,
        base_seed: int = 1234,
        cfg: MixConfig = MixConfig(),
        aug: AugmentCfg = AugmentCfg(),
        wave_cache_items: int = 512,
    ):
        super().__init__()
        self.rir = RIRBank(rir_root, split)
        self.speech = AudioIndex(speech_root, WaveCache(wave_cache_items))
        self.localn = AudioIndex(local_noises_root, self.speech.cache)   # share cache
        self.amb = AudioIndex(ambiences_root, self.speech.cache)
        self.cfg = cfg
        self.aug_cfg = aug
        self.epoch_size = epoch_size
        self.base_seed = int(base_seed)
        self._epoch = 0
        self._plan = StrictCountsPlanner(epoch_size, random.Random(self._seed_for("plan", 0))).build_plan()

    # reproducibility helpers
    def _seed_for(self, *tags) -> int:
        h = 0x9E3779B97F4A7C15
        for t in tags:
            if isinstance(t, (int, np.integer)):
                v = int(t)
            else:
                v = int.from_bytes(str(t).encode("utf-8"), "little", signed=False)
            h ^= v + 0x9E3779B97F4A7C15 + ((h << 6) & ((1<<64)-1)) + (h >> 2)
            h &= ((1<<64)-1)
        return h & 0x7FFFFFFF

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)
        rng = random.Random(self._seed_for("plan", epoch))
        self._plan = StrictCountsPlanner(self.epoch_size, rng).build_plan()

    def __len__(self): return self.epoch_size

    def __getitem__(self, idx: int):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker else 0
        seed = self._seed_for(self.base_seed, self._epoch, idx, wid)
        rng = random.Random(seed)

        n_sp = self._plan["n_speech"][idx]
        snr_name, snr_range = self._plan["snr_bins"][idx]
        snr_db = rng.uniform(*snr_range)

        # pick room + poses for speech
        room_meta, speech_poses = self.rir.sample_room_and_poses(
            rng, n_sp, min_sep_deg=self.cfg.min_sep_deg, enforce_radial_diversity=self.cfg.radial_diversity
        )
        room_id = room_meta["room"]["id"]
        rt60_s = float(room_meta["room"]["rt60_s"])

        T = int(round(self.cfg.dur_s * self.cfg.sr))
        device = torch.device("cpu")

        # build 4ch speech stem (and per-source stems if you want them)
        speech_stems = []
        az_list = []
        speech_ids = []
        for p in speech_poses:
            wav_path = self.speech.sample_path(rng)
            speech_ids.append(os.path.relpath(wav_path, self.speech.root))
            dry = self.speech.get_segment(wav_path, self.cfg.sr, T, rng)     # [T] mono
            dry = _set_rms(dry, self.cfg.target_rms)
            rirs_4 = self.rir.load_rirs_4(p).to(device)
            y4 = fft_conv_4ch_mono(dry.to(device), rirs_4)
            y4 = apply_per_mic_aug(y4, self.cfg.sr, rng, self.aug_cfg)
            speech_stems.append(y4)
            az_list.append(float(p.az_deg))

        speech_mix = torch.zeros(4, T, dtype=torch.float32, device=device)
        for y4 in speech_stems:
            speech_mix += y4

        # build noise
        noise_mix = torch.zeros_like(speech_mix)

        # diffuse ambience bed?
        amb_on = (rng.random() < self.cfg.ambience_prob_by_snr.get(snr_name, 0.7))
        amb_id = None
        if amb_on:
            amb_path = self.amb.sample_path(rng); amb_id = os.path.relpath(amb_path, self.amb.root)
            amb = self.amb.get_segment(amb_path, self.cfg.sr, T, rng)
            # sum K convolved poses from the SAME room for consistent reverb
            K = max(1, int(self.cfg.ambience_K_poses))
            # sample K poses (allow repeats)
            _, amb_poses = self.rir.sample_room_and_poses(rng, K, min_sep_deg=5.0, enforce_radial_diversity=False)
            for p in amb_poses:
                rirs_4 = self.rir.load_rirs_4(p).to(device)
                noise_mix += fft_conv_4ch_mono(amb.to(device), rirs_4)

        # local events (compound Poisson)
        lam = self.cfg.local_events_poisson_by_snr.get(snr_name, 1.0)
        n_local = np.random.default_rng(self._seed_for("poisson", seed)).poisson(lam=lam)
        n_local = int(np.clip(n_local, 0, 3))
        local_ids = []
        for _ in range(n_local):
            ln_path = self.localn.sample_path(rng); local_ids.append(os.path.relpath(ln_path, self.localn.root))
            full = self.localn.get_segment(ln_path, self.cfg.sr, T, rng)
            # crop shorter burst
            Ls = int(round(rng.uniform(*self.cfg.local_event_len_s_range) * self.cfg.sr))
            Ls = max(800, min(Ls, T//2))
            if full.numel() > Ls:
                st = _randint_exclusive(rng, 0, full.numel() - Ls + 1)
                full = full[st:st+Ls]
            # place at random time
            off = _randint_exclusive(rng, 0, T - full.numel() + 1)
            pad = torch.zeros(T, dtype=torch.float32)
            pad[off:off+full.numel()] = full
            # convolve through a random pose (keep same room)
            _, [p_loc] = self.rir.sample_room_and_poses(rng, 1, min_sep_deg=0.0, enforce_radial_diversity=False)
            rirs_4 = self.rir.load_rirs_4(p_loc).to(device)
            noise_mix += fft_conv_4ch_mono(pad.to(device), rirs_4)

        # scale noise to target SNR vs. speech_mix (over 4 mics)
        Es = (speech_mix**2).mean().item()
        En = (noise_mix**2).mean().item()
        if En < 1e-12:
            k = 0.0
        else:
            k = math.sqrt(Es / (En * (10.0 ** (snr_db/10.0))))
        mixture = speech_mix + noise_mix * k

        # soft peak guard (mild limiter)
        peak = mixture.abs().amax().item()
        if peak > 10 ** (-self.cfg.clip_guard_db/20):  # allow ~ -0.5 dBFS headroom
            mixture = mixture * ((10 ** (-self.cfg.clip_guard_db/20)) / peak)

        meta = {
            "n_speech": n_sp,
            "azimuths_deg": az_list,
            "snr_db": float(snr_db),
            "snr_bin": snr_name,
            "n_local_events": int(n_local),
            "has_ambience": bool(amb_on),
            "room_id": room_id,
            "rt60_s": rt60_s,
            "speech_files": speech_ids,
            "ambi_file": amb_id,
            "local_files": local_ids,
            "rir_pose_ids": [p.pose_id for p in speech_poses],
            "rir_meta": [p.meta_path for p in speech_poses],
        }
        return mixture, meta


#############################################
# ----------- DataLoader helpers -----------
#############################################

def worker_init_fn(worker_id: int):
    """Make numpy/random deterministic per worker."""
    info = torch.utils.data.get_worker_info()
    base_seed = info.seed  # torch already set a base seed
    random.seed(base_seed)
    np.random.seed(base_seed % (1<<32))

def collate_mix(batch: List[Tuple[torch.Tensor, Dict[str,Any]]]):
    xs = [b[0] for b in batch]
    metas = [b[1] for b in batch]
    x = torch.stack(xs, dim=0)  # [B, 4, T]
    return x, metas


#############################################
# ------------------ demo -------------------
#############################################

if __name__ == "__main__":
    
    """
    Example usage (adapt paths):

    python -m mix_batcher \
        --rir_root ./rir_bank \
        --split train \
        --speech_root ./speech_dataset \
        --local_root ./local_noises \
        --amb_root ./ambiances
    """

    import argparse, pprint, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--rir_root", default= './rir_bank'  )
    ap.add_argument("--split", default="train")
    ap.add_argument("--speech_root",    default = './datasets/LibriSpeech' )
    ap.add_argument("--local_root",     default = './datasets/noise'  )
    ap.add_argument("--amb_root",       default = './datasets/Ambiances' )
    ap.add_argument("--epoch_size", type=int, default=4000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=56)
    ap.add_argument("--test_type", type=str, default='savefile') # else 'test'
    ap.add_argument("--item_index", type=int, default=0, help="which item in the first batch to save")
    ap.add_argument("--out_wav", default="sample_mix.wav")
    ap.add_argument("--out_json", default="sample_mix.json")    
    args = ap.parse_args()

    if args.test_type == 'test':
        import soundfile as sf
        ds = OnTheFlyMixtureDataset(
            rir_root=args.rir_root, split=args.split,
            speech_root=args.speech_root, local_noises_root=args.local_root, ambiences_root=args.amb_root,
            epoch_size=args.epoch_size, base_seed=args.seed,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
            num_workers=args.num_workers, worker_init_fn=worker_init_fn,
            collate_fn=collate_mix, pin_memory=False,
        )

        n_batches = args.epoch_size // args.batch_size
        ds.set_epoch(0)

        # live counters
        c_nspeech = Counter()
        c_snr = Counter()
        c_amb = Counter()
        local_hist = Counter()
        az_hist = Counter()  # 72-bin azimuth histogram across speech sources

        pbar = tqdm(total=n_batches, desc="Epoch 0", unit="batch")
        for i, (x, metas) in enumerate(loader):
            # progress
            pbar.update(1)
            done = i + 1
            left = n_batches - done
            pbar.set_postfix_str(f"done {done}/{n_batches}, left {left}")

            for m in metas:
                c_nspeech[m["n_speech"]] += 1
                c_snr[m["snr_bin"]] += 1
                c_amb["amb_on" if m["has_ambience"] else "amb_off"] += 1
                local_hist[m["n_local_events"]] += 1
                # az coverage
                for az in m["azimuths_deg"]:
                    az_hist[quantize_az_deg_72(az)] += 1

        pbar.close()

        print("\n=== Observed distributions (one epoch) ===")
        print(f"batches: {n_batches}  items: {n_batches*args.batch_size}")
        print("n_speech:")
        for k in sorted(c_nspeech):
            tot = n_batches * args.batch_size
            print(f"  {k}: {c_nspeech[k]} ({c_nspeech[k]/tot*100:.2f}%)")

        print("SNR bins:")
        tot = sum(c_snr.values())
        for k in ["clean","medium","hard"]:
            v = c_snr[k]
            print(f"  {k}: {v} ({(v/tot*100 if tot else 0):.2f}%)")

        print("Ambience usage:")
        tot = sum(c_amb.values())
        for k in ["amb_on","amb_off"]:
            v = c_amb[k]
            print(f"  {k}: {v} ({(v/tot*100 if tot else 0):.2f}%)")

        print("Local events (count histogram):")
        for k in sorted(c_local := dict(c_local := local_hist).keys()):
            print(f"  {k}: {local_hist[k]}")

        # Azimuth coverage summary (top-5 bins by count)
        print("\nTop-5 azimuth bins (5° steps):")
        top5 = sorted(az_hist.items(), key=lambda kv: kv[1], reverse=True)[:5]
        for b, cnt in top5:
            print(f"  bin {b:02d} (deg ~ {b*5:3d}): {cnt}")
    elif args.test_type == 'savefile':

        ds = OnTheFlyMixtureDataset(
            rir_root=args.rir_root, split=args.split,
            speech_root=args.speech_root, local_noises_root=args.local_root, ambiences_root=args.amb_root,
            epoch_size=args.epoch_size, base_seed=args.seed,
        )
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False, drop_last=True,
            num_workers=args.num_workers, worker_init_fn=worker_init_fn,
            collate_fn=collate_mix, pin_memory=False,
        )

        # one epoch, first batch
        ds.set_epoch(0)
        x, metas = next(iter(loader))  # x: [B, 4, T]
        idx = max(0, min(args.item_index, x.shape[0]-1))
        mix = x[idx]                  # [4, T]
        meta = metas[idx]

        # save 4-ch WAV (T, C) for soundfile
        sr = ds.cfg.sr
        wav = mix.transpose(0, 1).cpu().numpy().astype(np.float32)  # [T, 4]
        sf.write(args.out_wav, wav, samplerate=sr, subtype="FLOAT")
        with open(args.out_json, "w") as f:
            json.dump(meta, f, indent=2)

        # pretty print meta
        print(f"Saved WAV:   {os.path.abspath(args.out_wav)}  (sr={sr}, shape={wav.shape})")
        print(f"Saved JSON:  {os.path.abspath(args.out_json)}")
        print("\n=== METADATA ===")
        # a bit of formatting for readability
        def fmt_list(v, maxn=6):
            return v if len(v) <= maxn else v[:maxn] + ["..."]
        pretty = dict(meta)  # shallow copy
        if isinstance(pretty.get("speech_files"), list):
            pretty["speech_files"] = fmt_list(pretty["speech_files"])
        if isinstance(pretty.get("local_files"), list):
            pretty["local_files"] = fmt_list(pretty["local_files"])
        print(json.dumps(pretty, indent=2))
