# BirdCLEF 2026 — ONNX Export + onnxruntime wheel cache
#
# Run this notebook ONLINE on Kaggle. It:
#   1. Rebuilds V2S / NFNet SED models exactly as in training
#   2. Loads each fold's best checkpoint
#   3. Exports to ONNX with dynamic batch (shape: [B, 1, 128, 313])
#   4. Downloads onnxruntime-gpu + onnxruntime wheels for offline install
#   5. Writes label2idx.json so the offline notebooks don't need a .pth
#
# After running, save /kaggle/working/onnx_assets as a Kaggle dataset and mount
# it in the offline submission / pseudo notebooks.

import os, json, subprocess, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ── CONFIG ────────────────────────────────────────────────────────────────────

OUT_DIR     = Path('/kaggle/working/onnx_assets')
MODELS_DIR  = OUT_DIR / 'models'
WHEELS_DIR  = OUT_DIR / 'wheels'
for d in (OUT_DIR, MODELS_DIR, WHEELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 234
SR          = 32_000
CLIP_SEC    = 5
CLIP_SAMP   = SR * CLIP_SEC      # 160_000
HOP_LENGTH  = 512
MEL_T       = CLIP_SAMP // HOP_LENGTH + 1  # 313 frames
MEL_N       = 128
OPSET       = 18   # 18+ avoids the dynamo down-conversion warning; ORT 1.16+ supports it

# List every checkpoint you want to use at inference time.
# arch ∈ {'v2s', 'nfnet'}.  The output filename is  f'{arch}__{stem}.onnx'.
CHECKPOINTS = [
    # V2-S folds
    ('/kaggle/input/datasets/berkeozdemir/lmlmmlml/best_sed_atthead_ns_fold0.pth', 'v2s'),
    ('/kaggle/input/datasets/berkeozdemir/lmlmmlml/best_sed_atthead_ns_fold1.pth', 'v2s'),
    ('/kaggle/input/datasets/berkeozdemir/lmlmmlml/best_sed_atthead_ns_fold2.pth', 'v2s'),
    ('/kaggle/input/datasets/berkeozdemir/lmlmmlml/best_sed_atthead_ns_fold3.pth', 'v2s'),
    ('/kaggle/input/datasets/berkeozdemir/lmlmmlml/best_sed_atthead_ns_fold4.pth', 'v2s'),
    # ECA-NFNet folds  — update paths as your nfnet checkpoints land
    # ('/kaggle/input/.../best_eca_nfnet_ns_fold0.pth', 'nfnet'),
    # ('/kaggle/input/.../best_eca_nfnet_ns_fold1.pth', 'nfnet'),
    # ('/kaggle/input/.../best_eca_nfnet_ns_fold2.pth', 'nfnet'),
    # ('/kaggle/input/.../best_eca_nfnet_ns_fold3.pth', 'nfnet'),
    # ('/kaggle/input/.../best_eca_nfnet_ns_fold4.pth', 'nfnet'),
]


# ── MODEL DEFINITIONS (byte-identical to training) ───────────────────────────

class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p); self.eps = eps
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(2, keepdim=True).pow(1.0 / self.p)


class AttHead(nn.Module):
    def __init__(self, in_chans, num_class=NUM_CLASSES, p=0.5):
        super().__init__()
        self.gem = GeMFreq()
        self.drop1 = nn.Dropout(p/2)
        self.fc1 = nn.Linear(in_chans, 1024)
        self.drop2 = nn.Dropout(p)
        self.att = nn.Conv1d(1024, num_class, 1)
        self.cla = nn.Conv1d(1024, num_class, 1)
    def forward(self, x):
        x = self.gem(x).squeeze(2).permute(0, 2, 1); x = self.drop1(x)
        x = F.relu(self.fc1(x)); x = self.drop2(x).permute(0, 2, 1)
        att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        val = self.cla(x)
        return (att * val).sum(dim=-1)


class V2SModel(nn.Module):
    """tf_efficientnetv2_s + AttHead. Input: mel spec (B, 1, 128, T)."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'tf_efficientnetv2_s', pretrained=False, num_classes=0,
            global_pool='', in_chans=1, drop_rate=0.0,
        )
        self.head = AttHead(self.backbone.num_features, NUM_CLASSES, p=0.5)
    def forward(self, x):
        return self.head(self.backbone(x))


class NFNetModel(nn.Module):
    """eca_nfnet_l0 (features_only) + AttHead. Input: mel spec (B, 1, 128, T)."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            'eca_nfnet_l0', pretrained=False, features_only=True,
            in_chans=1, drop_path_rate=0.0,
        )
        feat_ch = self.backbone.feature_info.channels()[-1]
        self.head = AttHead(feat_ch, NUM_CLASSES, p=0.5)
    def forward(self, x):
        feat = self.backbone(x)[-1]
        return self.head(feat)


def build_model(arch):
    if arch == 'v2s':   return V2SModel()
    if arch == 'nfnet': return NFNetModel()
    raise ValueError(f'unknown arch: {arch}')


# ── EXPORT ────────────────────────────────────────────────────────────────────

def export_one(ckpt_path, arch, out_path):
    ckpt = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    model = build_model(arch)
    # strict=False mirrors your training/submission code (tolerates optional buffers)
    missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=False)
    if missing:    print(f'    missing keys    : {len(missing)}  first={missing[:2]}')
    if unexpected: print(f'    unexpected keys : {len(unexpected)}  first={unexpected[:2]}')
    model.eval()

    dummy = torch.randn(1, 1, MEL_N, MEL_T)
    batch = torch.export.Dim('batch', min=1, max=64)
    torch.onnx.export(
        model, (dummy,), str(out_path),
        input_names=['spec'], output_names=['logits'],
        dynamic_shapes={'x': {0: batch}},
        opset_version=OPSET, do_constant_folding=True,
    )

    # Parity check: torch output vs onnxruntime output on the same dummy
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=['CPUExecutionProvider'])
    with torch.no_grad():
        t_out = model(dummy).numpy()
    o_out = sess.run(None, {'spec': dummy.numpy()})[0]
    max_abs = float(np.max(np.abs(t_out - o_out)))
    print(f'    parity max |Δ| = {max_abs:.2e}   (arch={arch})')
    assert max_abs < 1e-3, f'ONNX parity broken for {ckpt_path}: Δ={max_abs}'

    return ckpt.get('label2idx', None), float(ckpt.get('auc', -1)), int(ckpt.get('epoch', -1))


# ── WHEEL DOWNLOAD ────────────────────────────────────────────────────────────

def download_wheels():
    """Fetch wheels matching the CURRENT environment (= the Kaggle env the
    submission notebook will run in, as long as you don't change the base image)."""
    py = f'{sys.version_info.major}.{sys.version_info.minor}'
    print(f'\nDownloading onnxruntime wheels (python {py}) → {WHEELS_DIR}')
    for pkg in ('onnxruntime-gpu', 'onnxruntime'):
        cmd = ['pip', 'download', pkg, '-d', str(WHEELS_DIR)]
        print('  $', ' '.join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f'    WARNING: {pkg} download failed:\n{r.stderr[-400:]}')
        else:
            print(f'    ok')
    print('\nWheels present:')
    for w in sorted(WHEELS_DIR.glob('*.whl')):
        print(f'  {w.name}  ({w.stat().st_size / 1e6:.1f} MB)')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    manifest = []
    label2idx = None

    for ckpt_path, arch in CHECKPOINTS:
        if not Path(ckpt_path).exists():
            print(f'SKIP (not found): {ckpt_path}')
            continue
        stem = Path(ckpt_path).stem
        out_path = MODELS_DIR / f'{arch}__{stem}.onnx'
        print(f'\n[{arch}] {Path(ckpt_path).name}  →  {out_path.name}')
        l2i, auc, epoch = export_one(ckpt_path, arch, out_path)
        if label2idx is None and l2i is not None:
            label2idx = l2i
        manifest.append(dict(
            onnx=str(out_path.relative_to(OUT_DIR)),
            arch=arch,
            ckpt=str(ckpt_path),
            auc=auc, epoch=epoch,
        ))

    if label2idx is None:
        raise RuntimeError('No checkpoint provided label2idx — cannot build offline notebooks.')

    with open(OUT_DIR / 'label2idx.json', 'w') as f:
        json.dump(label2idx, f)
    with open(OUT_DIR / 'manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nlabel2idx: {len(label2idx)} classes  →  {OUT_DIR}/label2idx.json')
    print(f'manifest : {len(manifest)} models     →  {OUT_DIR}/manifest.json')

    download_wheels()

    print(f'\nAll assets in {OUT_DIR}')
    print('Next step: save /kaggle/working/onnx_assets as a Kaggle Dataset '
          'and mount it in the offline submission / pseudo notebooks.')


if __name__ == '__main__':
    main()
