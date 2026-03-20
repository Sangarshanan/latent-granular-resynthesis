"""
evaluation.py: Audio quality metrics for latent granular resynthesis.

Metrics:
    - FAD (Fréchet Audio Distance): distributional distance between pool and
      output embeddings via fadtk.  Lower = more pool-like timbre.
    - MFCC L2: Euclidean distance between mean MFCC vectors of output and pool.
      Lower = closer timbral match.
"""

import os
import tempfile
import numpy as np
import librosa
import soundfile as sf

from frechet_audio_distance import FrechetAudioDistance


SAMPLE_RATE = 24000


def mfcc_l2(
    output_audio: np.ndarray,
    pool_audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = 13,
) -> float:
    """
    L2 distance between the mean MFCC vectors of output and pool audio.

    Args:
        output_audio: 1-D float array — the resynthesised waveform.
        pool_audio:   1-D float array — the source pool waveform.
        sr:           Sample rate (default 24 kHz).
        n_mfcc:       Number of MFCC coefficients.

    Returns:
        Scalar L2 distance.  Lower → output timbre is closer to pool.
    """
    mfcc_out = librosa.feature.mfcc(y=output_audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_pool = librosa.feature.mfcc(y=pool_audio, sr=sr, n_mfcc=n_mfcc)

    mean_out = mfcc_out.mean(axis=1)
    mean_pool = mfcc_pool.mean(axis=1)

    return float(np.linalg.norm(mean_out - mean_pool))


def fad(
    output_audio: np.ndarray,
    pool_audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> float:
    """
    Fréchet Audio Distance between the output and pool audio.

    Uses the frechet_audio_distance package with an Encodec backbone.
    Audio arrays are written to temporary directories as wav files.

    Args:
        output_audio: 1-D float array — the resynthesised waveform.
        pool_audio:   1-D float array — the source pool waveform.
        sr:           Sample rate (default 24 kHz).

    Returns:
        FAD score (scalar).  Lower → output distribution is closer to pool.
    """
    if hasattr(output_audio, "detach"):
        output_audio = output_audio.detach().cpu().numpy()
    if hasattr(pool_audio, "detach"):
        pool_audio = pool_audio.detach().cpu().numpy()

    frechet = FrechetAudioDistance(
        model_name="encodec",
        sample_rate=sr,
        channels=1,
        verbose=False,
    )

    with tempfile.TemporaryDirectory() as baseline_dir, \
         tempfile.TemporaryDirectory() as eval_dir:

        sf.write(os.path.join(baseline_dir, "pool.wav"), pool_audio, sr)
        sf.write(os.path.join(eval_dir, "output.wav"), output_audio, sr)

        score = frechet.score(baseline_dir, eval_dir, dtype="float32")

    return float(score)


def evaluate(
    output_audio: np.ndarray,
    target_audio: np.ndarray,
    pool_audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    n_mfcc: int = 13,
) -> dict[str, float]:
    """
    Run all metrics and return a summary dict.

    Args:
        output_audio: 1-D float array — resynthesised waveform.
        target_audio: 1-D float array — original target waveform.
        pool_audio:   1-D float array — source pool waveform.
        sr:           Sample rate (default 24 kHz).
        n_mfcc:       Number of MFCC coefficients for MFCC L2.

    Returns:
        Dict with keys: "mfcc_l2", "snr_db", "fad".
    """
    return {
        "mfcc_l2": mfcc_l2(output_audio, pool_audio, sr, n_mfcc),
        "snr_db": snr(output_audio, target_audio),
        "fad": fad(output_audio, pool_audio, sr),
    }
