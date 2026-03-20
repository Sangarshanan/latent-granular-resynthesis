from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import sys
import numpy as np
import soundfile as sf
import torch
from pedalboard import Gain, Pedalboard, PitchShift
from transformers import EncodecModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from synthesis import decode_latents_to_audio, tokui_style_transfer_window  # noqa: E402
from utils import load_wav_mono  # noqa: E402

OUT = ROOT / "website" / "static" / "audio" / "examples"

SAMPLE_RATE = 24_000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _write_wav(path: Path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio).squeeze().astype(np.float32)
    sf.write(path, audio, sr, subtype="PCM_16")


def _source_wav(path: Path) -> np.ndarray:
    return load_wav_mono(str(path), target_sr=SAMPLE_RATE).squeeze().numpy()


AUGMENTATIONS = [
    Pedalboard([PitchShift(semitones=2)]),
    Pedalboard([PitchShift(semitones=-2)]),
    Pedalboard([PitchShift(semitones=7)]),
    Pedalboard([Gain(gain_db=6)]),
    Pedalboard([Gain(gain_db=-6)]),
    Pedalboard([PitchShift(semitones=4), Gain(gain_db=-3)]),
    Pedalboard([PitchShift(semitones=-4), Gain(gain_db=3)]),
]


@dataclass(frozen=True)
class WindowCfg:
    grain_size: int
    window_size: int
    stride: int
    hop: int
    augmentation: str  # "none" | "augmented"


def _run(
    model: EncodecModel,
    target: Path,
    pool: Path,
    cfg: WindowCfg,
    n_q: int = 8,
) -> np.ndarray:
    hybrid_latents, _ = tokui_style_transfer_window(
        model=model,
        target_file=str(target),
        pool_file=str(pool),
        n_q=n_q,
        window_size=cfg.window_size,
        hop=cfg.hop,
        grain_size=cfg.grain_size,
        stride=cfg.stride,
        pool_augmentations=(AUGMENTATIONS if cfg.augmentation == "augmented" else None),
    )
    audio = decode_latents_to_audio(model, hybrid_latents)
    return audio.squeeze()

def _run_cosine(
    model: EncodecModel,
    target: Path,
    pool: Path,
    n_q: int = 8,
) -> np.ndarray:
    from synthesis import tokui_style_transfer_cosine  # noqa: E402

    hybrid_latents, _ = tokui_style_transfer_cosine(
        model=model,
        target_file=str(target),
        pool_file=str(pool),
        n_q=n_q,
    )
    audio = decode_latents_to_audio(model, hybrid_latents)
    return audio.squeeze()


def main() -> None:
    print("Loading Encodec model…")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz", local_files_only=True)
    model.eval()
    model.to(DEVICE)

    percussion = ROOT / "wav24k" / "percussion"
    misc = ROOT / "wav24k" / "misc"
    instruments = ROOT / "wav24k" / "instruments"

    # --- Percussion: cosine works for noisy pool (kitchen pan) ---
    ex_dir = OUT / "percussion" / "cosine_kitchen_pan"
    target = percussion / "amen_mono_24k.wav"
    pool = misc / "kitchen_pan_mono_24k.wav"
    _write_wav(ex_dir / "target.wav", _source_wav(target))
    _write_wav(ex_dir / "pool.wav", _source_wav(pool))
    _write_wav(ex_dir / "output_cosine.wav", _run_cosine(model, target, pool))

    # --- Percussion: cosine fails on structured percussion (amen → tabla) ---
    ex_dir = OUT / "percussion" / "cosine_tabla_bad"
    target = percussion / "amen_mono_24k.wav"
    pool = percussion / "tabla.wav"
    _write_wav(ex_dir / "pool.wav", _source_wav(pool))
    _write_wav(ex_dir / "output_cosine.wav", _run_cosine(model, target, pool))

    # --- Percussion: windowed matching improves tabla ---
    ex_dir = OUT / "percussion" / "window_tabla"
    target = percussion / "amen_mono_24k.wav"
    pool = percussion / "tabla.wav"
    window_cfg = WindowCfg(grain_size=1, window_size=3, stride=1, hop=1, augmentation="none")
    _write_wav(ex_dir / "output_window.wav", _run(model, target, pool, window_cfg))

    # --- Percussion: grain size tradeoff (transient smearing) ---
    ex_dir = OUT / "percussion" / "grain_amen_to_tabla"
    target = percussion / "amen_mono_24k.wav"
    pool = percussion / "tabla.wav"
    g1 = WindowCfg(grain_size=1, window_size=3, stride=1, hop=1, augmentation="none")
    g5 = WindowCfg(grain_size=5, window_size=3, stride=1, hop=1, augmentation="none")
    _write_wav(ex_dir / "output_grain1.wav", _run(model, target, pool, g1))
    _write_wav(ex_dir / "output_grain5.wav", _run(model, target, pool, g5))

    # --- Instruments: augmentation effect (violin → sax, grain_size=5, stride=3) ---
    ex_dir = OUT / "instruments" / "aug_violin_to_sax"
    target = instruments / "violin.wav"
    pool = instruments / "sax.wav"
    cfg_none = WindowCfg(grain_size=5, window_size=1, stride=3, hop=1, augmentation="none")
    cfg_aug = WindowCfg(grain_size=5, window_size=1, stride=3, hop=1, augmentation="augmented")
    _write_wav(ex_dir / "output_none.wav", _run(model, target, pool, cfg_none))
    _write_wav(ex_dir / "output_augmented.wav", _run(model, target, pool, cfg_aug))

    # --- Instruments: bigger grains help ---
    ex_dir = OUT / "instruments" / "grain_violin_to_sax"
    target = instruments / "violin.wav"
    pool = instruments / "sax.wav"
    g1 = WindowCfg(grain_size=1)
    g5 = WindowCfg(grain_size=5)
    _write_wav(ex_dir / "output_grain1.wav", _run(model, target, pool, g1))
    _write_wav(ex_dir / "output_grain5.wav", _run(model, target, pool, g5))

    print(f"Done. Wrote examples under {OUT}")


if __name__ == "__main__":
    main()
