# BirdCLEF 2026 — OOF Pseudo-Label Generation (ONNX)
#
# Byte-identical pseudo generation logic to the PyTorch version, but inference
# runs via onnxruntime. Fold_id map, protected-file handling, TTA, thresholding
# and CSV output are unchanged.
#
# Supports SINGLE-ARCH or MULTI-ARCH ensembles. Set ARCHS to any subset of
# {'v2s', 'nfnet'}. For labeled soundscapes (fold_id >= 0) only the matching
# fold's model from each selected arch participates (OOF). For unlabeled
# soundscapes (fold_id == -1) every loaded session participates.
#
# Prereqs:
#   - onnx_export.py has been run with both v2s + nfnet checkpoints listed,
#     and its /kaggle/working/onnx_assets folder is mounted at ONNX_ROOT below.
#   - For each arch in ARCHS, exactly 5 ONNX paths (fold 0..4) must be supplied
#     in ONNX_FOLD_PATHS[arch].
#   - Both training runs (v2s + nfnet) must have used the same DUP_CSV_PATH
#     content and the same FOLD_SEED, so the protected-file set and fold
#     assignments are identical across architectures.

import json, subprocess, sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T


# ── CONFIG ────────────────────────────────────────────────────────────────────

ROOT      = Path('/kaggle/input/competitions/birdclef-2026')
ONNX_ROOT = Path('/kaggle/input/onnx-assets-v2s-nfnet')   # <-- edit to your mount

# Which architectures to ensemble. Any subset of {'v2s', 'nfnet'}.
#   ['v2s']            → v2s only          (output: pseudo_labels_v2s_oof.csv)
#   ['nfnet']          → nfnet only        (output: pseudo_labels_nfnet_oof.csv)
#   ['v2s', 'nfnet']   → ensemble of both  (output: pseudo_labels_v2s+nfnet_oof.csv)
ARCHS = ['v2s', 'nfnet']

# Fold models per arch — MUST be in fold order [fold 0, 1, 2, 3, 4].
# Only the entries for archs you list in ARCHS are read.
MODELS_DIR = ONNX_ROOT / 'models'
WHEELS_DIR = ONNX_ROOT / 'wheels'
ONNX_FOLD_PATHS = {
    'v2s': [
        MODELS_DIR / 'v2s__best_sed_atthead_ns_fold0.onnx',
        MODELS_DIR / 'v2s__best_sed_atthead_ns_fold1.onnx',
        MODELS_DIR / 'v2s__best_sed_atthead_ns_fold2.onnx',
        MODELS_DIR / 'v2s__best_sed_atthead_ns_fold3.onnx',
        MODELS_DIR / 'v2s__best_sed_atthead_ns_fold4.onnx',
    ],
    'nfnet': [
        MODELS_DIR / 'nfnet__best_eca_nfnet_ns_fold0.onnx',
        MODELS_DIR / 'nfnet__best_eca_nfnet_ns_fold1.onnx',
        MODELS_DIR / 'nfnet__best_eca_nfnet_ns_fold2.onnx',
        MODELS_DIR / 'nfnet__best_eca_nfnet_ns_fold3.onnx',
        MODELS_DIR / 'nfnet__best_eca_nfnet_ns_fold4.onnx',
    ],
}

OUT_TAG  = '+'.join(ARCHS)
OUT_PATH = Path(f'./pseudo_labels_{OUT_TAG}_oof.csv')

FOLD_SEED = 1215          # MUST match training SEED
N_FOLDS   = 5
MAX_TOTAL_FILES           = 2        # training compute_protected_files()
ALSO_PROTECT_SOLE_SC      = True     # training compute_protected_files()

# Duplicate cleanup CSV — must match training so train_counts feeding the
# protected-files computation is identical.
DUP_CSV_PATH = Path('/kaggle/input/datasets/vecihegrler/alsdkkldsaldks/duplicates.csv')

PRIMARY_LABEL_MIN_PROB = 0.5
TRIM_MIN_PROB          = 0.1

SR        = 32_000
CLIP_SEC  = 5
CLIP_SAMP = SR * CLIP_SEC
NUM_SEG   = 12
TTA_SHIFT = SR * 5 // 4

MEL_CFG = dict(
    n_fft=2048, hop_length=512, n_mels=128,
    f_min=20, f_max=16000,
    top_db=80.0,
)


# ── INSTALL onnxruntime ───────────────────────────────────────────────────────

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

    import os as _os
    current = _os.environ.get('LD_LIBRARY_PATH', '')
    _os.environ['LD_LIBRARY_PATH'] = ':'.join(lib_dirs + ([current] if current else []))

    preload_order = [
        'libcudart.so*', 'libnvrtc.so*', 'libcublasLt.so*', 'libcublas.so*',
        'libcufft.so*', 'libcurand.so*', 'libcusparse.so*', 'libcusolver.so*',
        'libnvJitLink.so*', 'libcudnn*.so*',
    ]
    loaded = []
    for d in lib_dirs:
        for pat in preload_order:
            for so in sorted(glob.glob(_os.path.join(d, pat))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(_os.path.basename(so))
                except OSError:
                    pass
    return loaded


def install_onnxruntime():
    try:
        import onnxruntime  # noqa: F401
        print(f'onnxruntime already importable: {onnxruntime.__version__}')
        return
    except Exception:
        pass
    if not WHEELS_DIR.exists():
        raise RuntimeError(f'No wheels dir: {WHEELS_DIR}')
    cmd = [sys.executable, '-m', 'pip', 'install',
           '--no-index', '--find-links', str(WHEELS_DIR), 'onnxruntime-gpu']
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


# ── SPECTROGRAM ───────────────────────────────────────────────────────────────

class Spectrogram(nn.Module):
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
        mel = self.amplitude_to_db(mel).unsqueeze(1)
        eps = 1e-6
        mean = mel.mean((2, 3), keepdim=True)
        std = mel.std((2, 3), keepdim=True)
        mel = (mel - mean) / (std + eps)
        norm_min = mel.amin(dim=(2, 3), keepdim=True)
        norm_max = mel.amax(dim=(2, 3), keepdim=True)
        return (mel - norm_min) / (norm_max - norm_min + eps)


# ── TRAINING DATA-ENG REPLICATION (must be byte-identical to training) ──────

def apply_duplicate_cleanup(train_df, dup_csv_path):
    """Byte-identical to training apply_duplicate_cleanup()."""
    if dup_csv_path is None or not Path(dup_csv_path).exists():
        print(f'  Duplicate cleanup SKIPPED (no {dup_csv_path})'); return train_df
    dup = pd.read_csv(dup_csv_path)
    dup['_species'] = dup['filename'].apply(lambda fn: fn.split('/')[0])
    to_drop = set(); n_conflict = n_redundant = 0
    for _, group in dup.groupby('hash'):
        if group['_species'].nunique() > 1:
            to_drop.update(group['filename']); n_conflict += len(group)
        else:
            canonical = sorted(group['filename'])[0]
            extras = set(group['filename']) - {canonical}
            to_drop.update(extras); n_redundant += len(extras)
    before = len(train_df)
    train_df = train_df[~train_df['filename'].isin(to_drop)].reset_index(drop=True)
    print(f'  Duplicate cleanup: {before} -> {len(train_df)} '
          f'({n_conflict} cross-species + {n_redundant} redundant)')
    return train_df


def compute_protected_files(sc_df, train_df, label2idx,
                            max_total_files=MAX_TOTAL_FILES,
                            also_protect_sole_sc=ALSO_PROTECT_SOLE_SC):
    """Byte-identical to training compute_protected_files()."""
    sp_to_scfiles = {}
    for _, r in sc_df.iterrows():
        for sp in str(r.primary_label).split(';'):
            sp = sp.strip()
            if sp and sp in label2idx:
                sp_to_scfiles.setdefault(sp, set()).add(r.filename)
    train_counts = train_df['primary_label'].value_counts()
    fragile = {sp for sp, files in sp_to_scfiles.items()
               if len(files) + int(train_counts.get(sp, 0)) <= max_total_files}
    to_protect = set(fragile)
    if also_protect_sole_sc:
        sole_sc = {sp for sp, files in sp_to_scfiles.items() if len(files) == 1}
        to_protect |= sole_sc
    protected = set()
    for sp in to_protect:
        protected.update(sp_to_scfiles[sp])
    n_sole_added = len(to_protect - fragile) if also_protect_sole_sc else 0
    print(f'  Fragile ≤{max_total_files}: {len(fragile)} species')
    print(f'  + Sole-sc species: {n_sole_added} more')
    print(f'  Total: {len(to_protect)} species, {len(protected)} protected files')
    return protected


def soundscape_folds(sc_df, n_splits=5, seed=FOLD_SEED, protected=None):
    """BYTE-IDENTICAL to training soundscape_folds().
    Key points vs the previous pseudo version:
      - seed = training SEED (1215), not 2
      - uses the per-site random OFFSET (rng.randint(n_splits)) — without
        this, the fold that fold N is 'val' in training will not match here
    """
    protected = protected or set()
    non = sc_df[~sc_df['filename'].isin(protected)].copy()
    non['site'] = non['filename'].apply(
        lambda fn: fn.split('_')[3] if len(fn.split('_')) > 3 else fn.split('_')[0]
    )
    rng = np.random.RandomState(seed)
    fm = {i: [] for i in range(n_splits)}
    for _, grp in non.groupby('site'):
        files = grp['filename'].unique().tolist()
        rng.shuffle(files)
        offset = rng.randint(n_splits)
        for i, fn in enumerate(files):
            fm[(i + offset) % n_splits].append(fn)
    return fm


def build_fold_id_map(label2idx):
    """Replicates the training data-eng pipeline end to end:
       1. load train.csv + dup cleanup
       2. load sc labels (same dedup as training)
       3. compute PROTECTED_FILES dynamically
       4. fold the NON-protected soundscapes with the same seed + offset logic
    Returns (fold_id_map, protected_files_set).
    """
    train_audio = pd.read_csv(ROOT / 'train.csv')
    train_audio = apply_duplicate_cleanup(train_audio, DUP_CSV_PATH)

    sc_labels = pd.read_csv(ROOT / 'train_soundscapes_labels.csv')
    sc_labels = sc_labels.drop_duplicates(subset=['filename', 'start', 'end'])

    protected = compute_protected_files(sc_labels, train_audio, label2idx)

    train_fold_map = soundscape_folds(sc_labels, N_FOLDS, FOLD_SEED, protected=protected)

    fold_id_map = {}
    for fold_i, files in train_fold_map.items():
        for fn in files:
            fold_id_map[fn.replace('.ogg', '')] = fold_i

    sc_dir = ROOT / 'train_soundscapes'
    for f in sorted(sc_dir.glob('*.ogg')):
        stem = f.stem
        if stem not in fold_id_map:
            fold_id_map[stem] = -1

    from collections import Counter
    dist = Counter(fold_id_map.values())
    print(f'\nFold distribution across {len(fold_id_map)} soundscapes:')
    for fold_i in sorted(dist.keys()):
        label = 'UNLABELED (ensemble)' if fold_i == -1 else f'fold {fold_i} (OOF single)'
        print(f'  {label}: {dist[fold_i]}')
    return fold_id_map, protected


# ── ONNX SESSIONS ─────────────────────────────────────────────────────────────

def build_providers():
    avail = ort.get_available_providers()
    providers = []
    if 'CUDAExecutionProvider' in avail:
        providers.append(('CUDAExecutionProvider', {'device_id': 0,
                                                    'arena_extend_strategy': 'kNextPowerOfTwo'}))
    providers.append('CPUExecutionProvider')
    return providers, avail


def load_sessions():
    """Returns sessions[arch][fold_id] -> InferenceSession.
    Arch key always present even if some folds are missing."""
    providers, avail = build_providers()
    print(f'ORT providers available: {avail}')
    print(f'ORT providers selected : {[p if isinstance(p, str) else p[0] for p in providers]}')
    sessions = {}
    total = 0
    for arch in ARCHS:
        if arch not in ONNX_FOLD_PATHS:
            raise RuntimeError(f'ARCHS contains "{arch}" but ONNX_FOLD_PATHS has no entry for it')
        sessions[arch] = {}
        for fold_i, p in enumerate(ONNX_FOLD_PATHS[arch]):
            if not Path(p).exists():
                print(f'  SKIP {arch} fold {fold_i} (not found): {p}')
                continue
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sessions[arch][fold_i] = ort.InferenceSession(str(p), sess_options=so, providers=providers)
            print(f'  {arch}  fold {fold_i}: {Path(p).name}')
            total += 1
    if total == 0:
        raise RuntimeError('No ONNX sessions loaded')
    print(f'  Loaded {total} sessions across {len(ARCHS)} arch(s): {ARCHS}')
    return sessions


def load_label2idx():
    with open(ONNX_ROOT / 'label2idx.json') as f:
        return json.load(f)


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


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def run():
    print(f'Device (torch / mel): {DEVICE}')

    label2idx  = load_label2idx()
    label_cols = [l for l, _ in sorted(label2idx.items(), key=lambda x: x[1])]
    print(f'  label2idx: {len(label2idx)} classes')

    print('\nBuilding fold_id map (replicating training data-eng pipeline)...')
    fold_id_map, protected_files = build_fold_id_map(label2idx)

    print('\nLoading ONNX sessions...')
    sessions = load_sessions()

    mel = Spectrogram(**MEL_CFG).to(DEVICE).eval()

    sc_dir = ROOT / 'train_soundscapes'
    files = sorted(sc_dir.glob('*.ogg'))
    print(f'\nTotal soundscape files: {len(files)}')

    all_row_ids, all_preds, all_fold_ids = [], [], []
    t0 = time()
    skipped_no_model  = 0
    skipped_protected = 0

    for i, fpath in enumerate(files):
        stem = fpath.stem

        if fpath.name in protected_files:
            skipped_protected += 1
            continue

        fold_id = fold_id_map.get(stem)
        if fold_id is None:
            skipped_no_model += 1
            continue

        row_ids = [f'{stem}_{(j+1)*CLIP_SEC}' for j in range(NUM_SEG)]

        segs       = load_segments(str(fpath), offset_samp=0).to(DEVICE)
        segs_shift = load_segments(str(fpath), offset_samp=TTA_SHIFT).to(DEVICE)

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=False):
                specs       = mel(segs).cpu().numpy()
                specs_shift = mel(segs_shift).cpu().numpy()

        # Pick which sessions are OOF for this file.
        #   fold_id == -1  → unlabeled  → ensemble EVERY loaded session
        #   fold_id >=  0  → labeled    → only the matching fold's model from
        #                                 each arch (those are the OOF ones)
        if fold_id == -1:
            active = [s for arch_sess in sessions.values() for s in arch_sess.values()]
        else:
            active = [sessions[arch][fold_id] for arch in ARCHS
                      if fold_id in sessions.get(arch, {})]
        if not active:
            skipped_no_model += 1
            continue

        probs       = np.mean([sigmoid_np(s.run(None, {'spec': specs})[0])       for s in active], axis=0)
        probs_shift = np.mean([sigmoid_np(s.run(None, {'spec': specs_shift})[0]) for s in active], axis=0)

        preds = (probs + probs_shift) / 2.0   # (12, n_classes)

        all_row_ids.extend(row_ids)
        all_preds.append(preds)
        all_fold_ids.extend([fold_id] * NUM_SEG)

        if (i + 1) % 500 == 0 or i == 0:
            elapsed = time() - t0
            fps = (i + 1) / elapsed
            remaining = (len(files) - i - 1) / fps if fps > 0 else 0
            print(f'  {i+1}/{len(files)}  {elapsed:.0f}s  '
                  f'~{remaining/60:.1f} min remaining')

    print(f'\n  Skipped {skipped_protected} protected files (excluded from pseudo pool by design)')
    if skipped_no_model > 0:
        print(f'  WARNING: {skipped_no_model} soundscapes skipped (no fold model)')

    all_preds = np.concatenate(all_preds, axis=0)

    df = pd.DataFrame(all_preds, columns=label_cols)
    df.insert(0, 'row_id',  all_row_ids)
    df.insert(1, 'fold_id', all_fold_ids)
    df['primary_label']      = df[label_cols].idxmax(axis=1)
    df['primary_label_prob'] = df[label_cols].max(axis=1)

    print(f'\nRaw OOF pseudo labels: {len(df)} rows')
    print(f'  prob distribution: min={df["primary_label_prob"].min():.3f}  '
          f'median={df["primary_label_prob"].median():.3f}  '
          f'mean={df["primary_label_prob"].mean():.3f}  '
          f'max={df["primary_label_prob"].max():.3f}')

    print(f'\nFold distribution in pseudos:')
    for fold_i in sorted(df['fold_id'].unique()):
        n = (df['fold_id'] == fold_i).sum()
        print(f'  fold {fold_i}: {n} rows')

    df_filtered = df[df['primary_label_prob'] >= PRIMARY_LABEL_MIN_PROB].copy()
    print(f'\nAfter filter (prob >= {PRIMARY_LABEL_MIN_PROB}): {len(df_filtered)} rows '
          f'({len(df_filtered)/len(df)*100:.1f}%)')

    probs = df_filtered[label_cols].values
    probs[probs < TRIM_MIN_PROB] = 0.0
    df_filtered[label_cols] = probs
    print(f'After trim (< {TRIM_MIN_PROB} -> 0): '
          f'avg spp/chunk = {(probs > 0).sum(axis=1).mean():.1f}')

    raw_path = str(OUT_PATH).replace('.csv', '_raw.csv')
    df.to_csv(raw_path, index=False)
    print(f'\nSaved RAW: {raw_path}')

    df_filtered.to_csv(OUT_PATH, index=False)
    elapsed = time() - t0
    print(f'Saved filtered: {OUT_PATH}  ({len(df_filtered)} rows, {elapsed/60:.1f} min)')


if __name__ == '__main__':
    run()
