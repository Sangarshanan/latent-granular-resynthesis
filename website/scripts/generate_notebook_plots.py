from __future__ import annotations

from pathlib import Path

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import EncodecModel

import matplotlib

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tmp" / "mplconfig"))
sys.path.insert(0, str(ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from synthesis import extract_latents, tokui_style_transfer_window  # noqa: E402
from utils import load_wav_mono  # noqa: E402

OUT_DIR = ROOT / "website" / "static" / "plots" / "notebook"

SR = 24_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_fig(name: str) -> Path:
    path = OUT_DIR / name
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def _pca2(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = X.astype(np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    comps = Vt[:2].T
    Y = X @ comps
    var = (S**2) / max(1, (X.shape[0] - 1))
    evr = var[:2] / max(1e-12, var.sum())
    return Y, evr


def _subsample(z_TD: torch.Tensor, max_points: int = 1500) -> np.ndarray:
    z = z_TD.detach().cpu().numpy()
    if z.shape[0] <= max_points:
        return z
    idxs = np.linspace(0, z.shape[0] - 1, max_points).astype(int)
    return z[idxs]


def main() -> None:
    _ensure_out()

    target_file = "wav24k/percussion/amen_mono_24k.wav"
    pool_file = "wav24k/percussion/tabla.wav"
    n_q = 32
    window_size = 3
    grain_size = 1
    hop = 1
    stride = 1

    print("Loading Encodec model…")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    model.eval()
    model.to(DEVICE)

    print("Running windowed style transfer…")
    hybrid, idx = tokui_style_transfer_window(
        model=model,
        target_file=target_file,
        pool_file=pool_file,
        n_q=n_q,
        window_size=window_size,
        hop=hop,
        grain_size=grain_size,
        stride=stride,
    )

    print("Encoding latents for analysis…")
    target_latents, pool_latents = extract_latents(
        model=model, target_file=target_file, pool_file=pool_file, n_q=n_q
    )

    idx = idx.clamp(0, pool_latents.shape[0] - 1)
    matched_pool = pool_latents[idx]

    # Waveforms
    target_audio = load_wav_mono(str(ROOT / target_file), target_sr=SR).squeeze().numpy()
    pool_audio = load_wav_mono(str(ROOT / pool_file), target_sr=SR).squeeze().numpy()

    from synthesis import decode_latents_to_audio  # local import keeps module graph small

    out_audio = decode_latents_to_audio(model, hybrid).squeeze()
    t_tgt = np.arange(len(target_audio)) / SR
    t_pool = np.arange(len(pool_audio)) / SR
    t_out = np.arange(len(out_audio)) / SR

    fig, ax = plt.subplots(3, 1, figsize=(12, 6), sharex=False)
    ax[0].plot(t_tgt, target_audio, lw=0.8)
    ax[0].set_title("Target waveform (amen)")
    ax[1].plot(t_pool, pool_audio, lw=0.8)
    ax[1].set_title("Pool waveform (tabla)")
    ax[2].plot(t_out, out_audio, lw=0.8)
    ax[2].set_title("Output waveform (windowed match)")
    for a in ax:
        a.set_xlabel("Time (s)")
        a.set_ylabel("Amplitude")
    plt.tight_layout()
    _save_fig("amen_tabla_waveforms.png")

    # Similarity over time
    t_norm = F.normalize(target_latents, dim=-1)
    m_norm = F.normalize(matched_pool, dim=-1)
    sim_t = (t_norm * m_norm).sum(dim=-1).detach().cpu().numpy()
    # Encodec 24k: 320 hop in samples for 75 Hz frame rate
    frame_rate = SR / 320.0
    t_frames = np.arange(sim_t.shape[0]) / frame_rate

    plt.figure(figsize=(12, 2.6))
    plt.plot(t_frames, sim_t, lw=1.0)
    plt.ylim(-1.0, 1.0)
    plt.title("Cosine similarity over time (target latents vs matched pool latents)")
    plt.xlabel("Time (s)")
    plt.ylabel("Cosine sim")
    plt.tight_layout()
    _save_fig("amen_tabla_similarity_over_time.png")


    # PCA scatter
    Z_tgt = _subsample(target_latents)
    Z_pool = _subsample(pool_latents)
    Z_match = _subsample(matched_pool)
    Z_out = _subsample(hybrid)

    X = np.vstack([Z_tgt, Z_pool, Z_match, Z_out])
    Y, evr = _pca2(X)

    n1, n2, n3, n4 = map(len, [Z_tgt, Z_pool, Z_match, Z_out])
    s1 = slice(0, n1)
    s2 = slice(n1, n1 + n2)
    s3 = slice(n1 + n2, n1 + n2 + n3)
    s4 = slice(n1 + n2 + n3, n1 + n2 + n3 + n4)

    plt.figure(figsize=(8, 6))
    plt.scatter(Y[s1, 0], Y[s1, 1], s=6, alpha=0.35, label="target (amen)")
    plt.scatter(Y[s2, 0], Y[s2, 1], s=6, alpha=0.35, label="pool (tabla)")
    plt.scatter(Y[s3, 0], Y[s3, 1], s=8, alpha=0.35, label="matched-pool frames")
    plt.title(f"PCA of latent frames (EVR: {evr[0]:.2f}, {evr[1]:.2f})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(markerscale=2, frameon=False)
    plt.tight_layout()
    _save_fig("amen_tabla_latents_pca.png")

    print(f"Done. Wrote plots under {OUT_DIR}")


if __name__ == "__main__":
    main()
