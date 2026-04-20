# BirdCLEF 2026 — ONNX Submission (offline)
#
# Identical pipeline to the PyTorch submission notebook, but inference runs via
# onnxruntime-gpu. Mel spectrogram, TTA, postprocessing, time smoothing and
# prior blending are untouched.
#
# To switch between v2s-only / nfnet-only / combined ensemble: just edit
# ONNX_PATHS below. The submission treats each entry as one "fold" in the
# ensemble — arch is irrelevant once exported.
#
# Prereqs:
#   - Run onnx_export.py online, save /kaggle/working/onnx_assets as a dataset
#   - Mount that dataset at ONNX_ROOT below

import json, os, subprocess, sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


# ── CONFIG ────────────────────────────────────────────────────────────────────

ROOT       = Path('/kaggle/input/competitions/birdclef-2026')
OUT_PATH   = Path('./submission.csv')

# Point at the mounted onnx_assets dataset
ONNX_ROOT  = Path('/kaggle/input/onnx-assets-v2s-nfnet')   # <-- edit to match mount
MODELS_DIR = ONNX_ROOT / 'models'
WHEELS_DIR = ONNX_ROOT / 'wheels'

# ── Pick which ONNX sessions enter the ensemble ──────────────────────────────
# Toggle between three configurations by commenting/uncommenting lines below.
ONNX_PATHS = [
    # V2-S only
    MODELS_DIR / 'v2s__best_sed_atthead_ns_fold0.onnx',
    MODELS_DIR / 'v2s__best_sed_atthead_ns_fold1.onnx',
    MODELS_DIR / 'v2s__best_sed_atthead_ns_fold2.onnx',
    MODELS_DIR / 'v2s__best_sed_atthead_ns_fold3.onnx',
    MODELS_DIR / 'v2s__best_sed_atthead_ns_fold4.onnx',

    # NFNet only (uncomment these and comment v2s block above for nfnet-only)
    # MODELS_DIR / 'nfnet__best_eca_nfnet_ns_fold0.onnx',
    # MODELS_DIR / 'nfnet__best_eca_nfnet_ns_fold1.onnx',
    # MODELS_DIR / 'nfnet__best_eca_nfnet_ns_fold2.onnx',
    # MODELS_DIR / 'nfnet__best_eca_nfnet_ns_fold3.onnx',
    # MODELS_DIR / 'nfnet__best_eca_nfnet_ns_fold4.onnx',
]

PRIOR_WEIGHT = 0.05

# ── POLISH KNOBS ─────────────────────────────────────────────────────────────
USE_MAX_BOOST_MULT = True
USE_MAX_BOOST_ADD  = False
USE_MINMAX_PER_COL = False
POWER_P            = 1.0
BOOST_ALPHA        = 0.5

SR          = 32_000
CLIP_SEC    = 5
CLIP_SAMP   = SR * CLIP_SEC
NUM_SEG     = 12
TTA_SHIFT   = SR * 5 // 4
NUM_CLASSES = 234

MEL_CFG = dict(
    n_fft=2048, hop_length=512, n_mels=128,
    f_min=20, f_max=16000,
    top_db=80.0,
)

SMOOTH_EVENT   = np.array([0.20, 0.60, 0.20])
SMOOTH_TEXTURE = np.array([0.35, 0.30, 0.35])


# ── INSTALL onnxruntime FROM LOCAL WHEELS ────────────────────────────────────

def _preload_nvidia_cuda_libs():
    """onnxruntime-gpu (CUDA 12 build) expects libcublasLt.so.12 etc. on the
    loader path, but Kaggle ships CUDA via pip packages (nvidia-cublas-cu12,
    nvidia-cudnn-cu12, ...) that aren't on LD_LIBRARY_PATH. We discover them
    in site-packages, prepend them to LD_LIBRARY_PATH, and ctypes-preload the
    key .so files so any later dlopen() from ORT resolves.
    Must run BEFORE `import onnxruntime`.
    """
    import sys, pathlib, ctypes, glob
    roots = []
    for sp in sys.path:
        p = pathlib.Path(sp) / 'nvidia'
        if p.exists():
            roots.append(p)
    lib_dirs = []
    for r in roots:
        for sub in r.iterdir():
            lib = sub / 'lib'
            if lib.exists():
                lib_dirs.append(str(lib))
    if not lib_dirs:
        return []

    current = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_dirs + ([current] if current else []))

    # ctypes-preload (order matters: dependencies first)
    preload_order = [
        'libcudart.so*', 'libnvrtc.so*', 'libcublasLt.so*', 'libcublas.so*',
        'libcufft.so*', 'libcurand.so*', 'libcusparse.so*', 'libcusolver.so*',
        'libnvJitLink.so*', 'libcudnn*.so*',
    ]
    loaded = []
    for d in lib_dirs:
        for pat in preload_order:
            for so in sorted(glob.glob(os.path.join(d, pat))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(so))
                except OSError:
                    pass
    return loaded


def install_onnxruntime():
    """Install onnxruntime-gpu (fallback: onnxruntime) from wheels dataset.
    Skipped entirely if an onnxruntime build is already importable."""
    try:
        import onnxruntime  # noqa: F401
        print(f'onnxruntime already importable: {onnxruntime.__version__}')
        return
    except Exception:
        pass
    if not WHEELS_DIR.exists():
        raise RuntimeError(f'No wheels dir: {WHEELS_DIR}')
    cmd = [sys.executable, '-m', 'pip', 'install',
           '--no-index', '--find-links', str(WHEELS_DIR),
           'onnxruntime-gpu']
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print('onnxruntime-gpu install failed, trying onnxruntime (CPU):')
        print(r.stderr[-400:])
        cmd[-1] = 'onnxruntime'
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'Both installs failed: {r.stderr[-400:]}')
    print(f'onnxruntime installed from {WHEELS_DIR}')


install_onnxruntime()
_preloaded = _preload_nvidia_cuda_libs()
if _preloaded:
    print(f'Preloaded {len(_preloaded)} CUDA libs (cublasLt, cudnn, ...)')
else:
    print('No nvidia-* CUDA libs found in site-packages — ORT will use CPU only')
import onnxruntime as ort

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── TAXONOMY ──────────────────────────────────────────────────────────────────

def load_taxonomy(label_cols):
    tax = pd.read_csv(ROOT / 'taxonomy.csv')
    tax['primary_label'] = tax['primary_label'].astype(str)
    texture_classes = set(
        tax[tax['class_name'].isin(['Insecta', 'Amphibia'])]['primary_label']
    )
    return np.array([l in texture_classes for l in label_cols], dtype=bool)


# ── SPECTROGRAM ───────────────────────────────────────────────────────────────

class Spectrogram(nn.Module):
    """Must match training Spectrogram exactly."""
    def __init__(self, n_fft=2048, hop_length=512, n_mels=128,
                 f_min=20, f_max=16000, top_db=80.0, **_):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SR, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, f_min=f_min, f_max=f_max,
            normalized=True, power=2.0, center=True,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=top_db)

    def forward(self, x):
        mel = self.mel_transform(x.float())
        mel = self.amplitude_to_db(mel)
        mel = mel.unsqueeze(1)
        eps = 1e-6
        mean = mel.mean((2, 3), keepdim=True)
        std = mel.std((2, 3), keepdim=True)
        mel = (mel - mean) / (std + eps)
        norm_min = mel.amin(dim=(2, 3), keepdim=True)
        norm_max = mel.amax(dim=(2, 3), keepdim=True)
        return (mel - norm_min) / (norm_max - norm_min + eps)


# ── ONNX SESSIONS ─────────────────────────────────────────────────────────────

def build_providers():
    avail = ort.get_available_providers()
    providers = []
    if 'CUDAExecutionProvider' in avail:
        providers.append(('CUDAExecutionProvider', {'device_id': 0,
                                                    'arena_extend_strategy': 'kNextPowerOfTwo'}))
    providers.append('CPUExecutionProvider')
    return providers, avail


def load_onnx_sessions(paths):
    providers, avail = build_providers()
    print(f'ORT providers available: {avail}')
    print(f'ORT providers selected : {[p if isinstance(p, str) else p[0] for p in providers]}')

    sess_list = []
    for p in paths:
        if not Path(p).exists():
            print(f'  SKIP (not found): {p}')
            continue
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(str(p), sess_options=so, providers=providers)
        sess_list.append(sess)
        print(f'  Loaded ONNX: {Path(p).name}')
    if not sess_list:
        raise RuntimeError('No ONNX sessions loaded')
    print(f'  Ensemble: {len(sess_list)} sessions')
    return sess_list


def load_label2idx():
    with open(ONNX_ROOT / 'label2idx.json') as f:
        return json.load(f)


# ── SITE / HOUR PRIORS ────────────────────────────────────────────────────────

def build_prior_tables(label_cols):
    sc  = pd.read_csv(ROOT / 'train_soundscapes_labels.csv')
    sc  = sc.drop_duplicates(subset=['filename', 'start', 'end'])
    label_to_idx = {l: i for i, l in enumerate(label_cols)}
    n  = len(label_cols)

    def parse_meta(filename):
        parts = filename.split('_')
        site  = parts[3] if len(parts) > 3 else 'UNK'
        hour  = int(parts[5][:2]) if len(parts) > 5 else -1
        return site, hour

    rows = []
    for _, r in sc.iterrows():
        site, hour = parse_meta(r['filename'])
        y = np.zeros(n, dtype=np.float32)
        for sp in str(r['primary_label']).split(';'):
            sp = sp.strip()
            if sp in label_to_idx:
                y[label_to_idx[sp]] = 1.0
        rows.append((site, hour, y))

    global_p = np.mean([r[2] for r in rows], axis=0).astype(np.float32)

    sites = {}
    for site, hour, y in rows:
        sites.setdefault(site, []).append(y)
    site_p = {s: np.mean(v, axis=0).astype(np.float32) for s, v in sites.items()}
    site_n = {s: len(v) for s, v in sites.items()}

    hours = {}
    for site, hour, y in rows:
        if hour < 0: continue
        hours.setdefault(hour, []).append(y)
    hour_p = {h: np.mean(v, axis=0).astype(np.float32) for h, v in hours.items()}
    hour_n = {h: len(v) for h, v in hours.items()}

    print(f'  Prior tables: {len(site_p)} sites, {len(hour_p)} hours, '
          f'global_p mean={global_p.mean():.4f}')
    return dict(global_p=global_p, site_p=site_p, site_n=site_n,
                hour_p=hour_p, hour_n=hour_n)


def get_prior(site, hour, tables, smooth_k=8.0):
    p = tables['global_p'].copy()
    if hour >= 0 and hour in tables['hour_p']:
        nh = tables['hour_n'][hour]
        wh = nh / (nh + smooth_k)
        p  = wh * tables['hour_p'][hour] + (1 - wh) * p
    if site in tables['site_p']:
        ns = tables['site_n'][site]
        ws = ns / (ns + smooth_k)
        p  = ws * tables['site_p'][site] + (1 - ws) * p
    return p.astype(np.float32)


def parse_filename_meta(stem):
    parts = stem.split('_')
    site  = parts[3] if len(parts) > 3 else 'UNK'
    hour  = int(parts[5][:2]) if len(parts) > 5 else -1
    return site, hour


# ── POSTPROCESSING ────────────────────────────────────────────────────────────

def postprocessing(preds, top=1):
    N, F = preds.shape
    preds_3d = preds.reshape((N // NUM_SEG, NUM_SEG, F))
    top_k = np.sort(preds_3d, axis=1)[:, -top:]
    mean_top = np.mean(top_k, axis=1, keepdims=True)
    preds_3d = preds_3d * mean_top
    return preds_3d.reshape((N, F))


def additive_max_boost(preds, alpha=0.5):
    N, F = preds.shape
    preds_3d = preds.reshape((N // NUM_SEG, NUM_SEG, F))
    max_per_file  = preds_3d.max(axis=1, keepdims=True)
    mean_per_file = preds_3d.mean(axis=1, keepdims=True)
    preds_3d = preds_3d + alpha * (max_per_file - mean_per_file)
    return preds_3d.reshape((N, F))


def minmax_per_col(preds):
    col_min = preds.min(axis=0, keepdims=True)
    col_max = preds.max(axis=0, keepdims=True)
    return (preds - col_min) / (col_max - col_min + 1e-8)


def power_scale(preds, p=1.0):
    if p == 1.0: return preds
    return np.power(np.clip(preds, 0.0, None), p)


def time_smooth(preds, is_texture):
    if NUM_SEG <= 2: return preds
    def smooth(p, w):
        pad = np.pad(p, ((1, 1), (0, 0)), mode='edge')
        return w[0] * pad[:-2] + w[1] * pad[1:-1] + w[2] * pad[2:]
    result = preds.copy()
    if is_texture.any():
        result[:, is_texture]  = smooth(preds[:, is_texture],  SMOOTH_TEXTURE)
    if (~is_texture).any():
        result[:, ~is_texture] = smooth(preds[:, ~is_texture], SMOOTH_EVENT)
    return result


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


# ── AUDIO ─────────────────────────────────────────────────────────────────────

def load_segments(filepath, offset_samp=0):
    try:
        wav, sr = torchaudio.load(str(filepath))
        wav = wav.mean(0).float()
        if sr != SR:
            wav = torchaudio.functional.resample(wav, sr, SR)
        target = SR * 60
        if wav.shape[0] < target:
            pad = torch.zeros(target); pad[:wav.shape[0]] = wav; wav = pad
        else:
            wav = wav[:target]
        if offset_samp > 0:
            wav = torch.roll(wav, offset_samp)
        return wav.reshape(NUM_SEG, CLIP_SAMP)
    except Exception as e:
        print(f'  Warning: {Path(filepath).name}: {e}')
        return torch.zeros(NUM_SEG, CLIP_SAMP)


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_inference():
    print(f'Device (torch / mel): {DEVICE}')

    print('Loading ONNX sessions...')
    sessions = load_onnx_sessions(ONNX_PATHS)

    label2idx  = load_label2idx()
    label_cols = [l for l, _ in sorted(label2idx.items(), key=lambda x: x[1])]
    n_classes  = len(label_cols)
    print(f'  label2idx: {n_classes} classes')

    is_texture = load_taxonomy(label_cols)
    print(f'  Texture classes: {is_texture.sum()}  Event classes: {(~is_texture).sum()}')

    mel = Spectrogram(**MEL_CFG).to(DEVICE).eval()

    print('Building site/hour prior tables...')
    prior_tables = build_prior_tables(label_cols)

    test_dir = ROOT / 'test_soundscapes'
    files    = sorted(test_dir.glob('*.ogg'))
    if len(files) == 0:
        sample = pd.read_csv(ROOT / 'sample_submission.csv')
        stems  = sorted(set('_'.join(r.split('_')[:-1]) for r in sample['row_id']))
        files  = [test_dir / f'{s}.ogg' for s in stems]
        print(f'Test folder empty — dry run on {len(files)} files')
    print(f'Test files: {len(files)}')

    all_row_ids, all_preds = [], []
    t0 = time()

    for i, fpath in enumerate(files):
        stem    = Path(fpath).stem
        row_ids = [f'{stem}_{(j+1)*CLIP_SEC}' for j in range(NUM_SEG)]

        segs       = load_segments(str(fpath), offset_samp=0).to(DEVICE)
        segs_shift = load_segments(str(fpath), offset_samp=TTA_SHIFT).to(DEVICE)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                specs       = mel(segs).cpu().numpy()
                specs_shift = mel(segs_shift).cpu().numpy()

        fold_probs, fold_probs_shift = [], []
        for sess in sessions:
            logits       = sess.run(None, {'spec': specs})[0]
            logits_shift = sess.run(None, {'spec': specs_shift})[0]
            fold_probs.append(sigmoid_np(logits))
            fold_probs_shift.append(sigmoid_np(logits_shift))

        cnn_preds = (np.mean(fold_probs, axis=0) +
                     np.mean(fold_probs_shift, axis=0)) / 2.0

        all_row_ids.extend(row_ids)
        all_preds.append(cnn_preds)

        if (i + 1) % 200 == 0 or i == 0:
            elapsed   = time() - t0
            fps       = (i + 1) / elapsed
            remaining = (len(files) - i - 1) / fps if fps > 0 else 0
            print(f'  {i+1}/{len(files)}  {elapsed:.0f}s  '
                  f'~{remaining/60:.1f} min remaining')

    all_preds = np.concatenate(all_preds, axis=0)

    # ── POLISH STACK (identical to PyTorch notebook) ─────────────────────
    if USE_MAX_BOOST_MULT:
        all_preds = postprocessing(all_preds, top=1)
        print(f'  polish: multiplicative max-boost applied')
    if USE_MAX_BOOST_ADD:
        all_preds = additive_max_boost(all_preds, alpha=BOOST_ALPHA)
        print(f'  polish: additive max-boost applied (alpha={BOOST_ALPHA})')
    if USE_MINMAX_PER_COL:
        all_preds = minmax_per_col(all_preds)
        print(f'  polish: per-column min-max normalization applied')
    if POWER_P != 1.0:
        all_preds = power_scale(all_preds, p=POWER_P)
        print(f'  polish: power scaling applied (p={POWER_P})')

    final_preds = []
    for fi in range(len(files)):
        stem = Path(files[fi]).stem
        site, hour = parse_filename_meta(stem)
        preds = all_preds[fi * NUM_SEG : (fi + 1) * NUM_SEG]
        preds = time_smooth(preds, is_texture)
        prior = get_prior(site, hour, prior_tables)
        preds = (1 - PRIOR_WEIGHT) * preds + PRIOR_WEIGHT * prior[None, :]
        final_preds.append(preds)
    all_preds = np.concatenate(final_preds, axis=0)

    sub = pd.DataFrame(all_preds, columns=label_cols)
    sub.insert(0, 'row_id', all_row_ids)

    sample = pd.read_csv(ROOT / 'sample_submission.csv')
    assert list(sub.columns) == list(sample.columns), \
        f'Column mismatch!\nExpected: {list(sample.columns)[:5]}\nGot: {list(sub.columns)[:5]}'

    sub.to_csv(OUT_PATH, index=False)
    elapsed = time() - t0
    print(f'\nDone in {elapsed/60:.1f} min')
    print(f'Submission: {sub.shape}  '
          f'predictions [{all_preds.min():.4f}, {all_preds.max():.4f}]')
    print(f'Ensemble size: {len(sessions)} sessions')
    print(f'Prior blending: ON (w={PRIOR_WEIGHT})')
    print(f'Polish: mult={USE_MAX_BOOST_MULT} add={USE_MAX_BOOST_ADD} '
          f'minmax={USE_MINMAX_PER_COL} power_p={POWER_P} alpha={BOOST_ALPHA}')


if __name__ == '__main__':
    run_inference()
