"""
Evaluate tokui_style_transfer_window across all pairwise combinations of
audio files in wav24k/percussion and wav24k/instruments.

Parameter grid:
    - grain_size:        1–5
    - window_size:       1–5
    - stride:            1–3
    - hop:               1–3
    - pool_augmentations: None  vs  augmentation chain

All results are written to evaluation_results.csv.

Usage:
    python grid_evaluation.py
"""

import csv
import os
import itertools

import torch
from transformers import EncodecModel
from pedalboard import Pedalboard, PitchShift, Gain

from synthesis import (
    tokui_style_transfer_window,
    decode_latents_to_audio,
    SAMPLE_RATE,
)
from utils import load_wav_mono
from evaluation import mfcc_l2, fad

# ── Directories ──────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(__file__)
PERCUSSION_DIR = os.path.join(BASE_DIR, "wav24k", "percussion")
INSTRUMENTS_DIR = os.path.join(BASE_DIR, "wav24k", "instruments")
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
OUTPUT_CSV = os.path.join(BASE_DIR, "evaluation_results.csv")

# ── Hyper-parameter grid ─────────────────────────────────────────────────────

N_Q = 8
GRAIN_SIZES = [1, 2, 3, 4, 5]
WINDOW_SIZES = [1, 2, 3, 4, 5]
STRIDES = [1, 2, 3]
HOPS = [1, 2, 3]

# ── Augmentations ────────────────────────────────────────────────────────────

AUGMENTATIONS = [
    Pedalboard([PitchShift(semitones=2)]),
    Pedalboard([PitchShift(semitones=-2)]),
    Pedalboard([PitchShift(semitones=7)]),
    Pedalboard([Gain(gain_db=6)]),
    Pedalboard([Gain(gain_db=-6)]),
    Pedalboard([PitchShift(semitones=4), Gain(gain_db=-3)]),
    Pedalboard([PitchShift(semitones=-4), Gain(gain_db=3)]),
]

AUGMENTATION_CONFIGS = [
    (None, "none"),
    (AUGMENTATIONS, "augmented"),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "category", "target", "pool",
    "grain_size", "window_size", "stride", "hop",
    "augmentation", "mfcc_l2", "fad",
]


def collect_audio_files(directory: str) -> list[str]:
    files = []
    for fname in sorted(os.listdir(directory)):
        if os.path.splitext(fname)[1].lower() in AUDIO_EXTENSIONS:
            files.append(os.path.join(directory, fname))
    return files


def run_single(
    model,
    target_file: str,
    pool_file: str,
    n_q: int,
    window_size: int,
    grain_size: int,
    stride: int,
    hop: int,
    pool_augmentations: list | None,
) -> dict[str, float]:
    hybrid_latents, _ = tokui_style_transfer_window(
        model=model,
        target_file=target_file,
        pool_file=pool_file,
        n_q=n_q,
        window_size=window_size,
        grain_size=grain_size,
        stride=stride,
        hop=hop,
        pool_augmentations=pool_augmentations,
    )

    output_audio = decode_latents_to_audio(model, hybrid_latents)
    output_1d = output_audio.squeeze()

    pool_1d = load_wav_mono(pool_file).squeeze().numpy()

    return {
        "mfcc_l2": mfcc_l2(output_1d, pool_1d, sr=SAMPLE_RATE),
        "fad": fad(output_1d, pool_1d, sr=SAMPLE_RATE),
    }


def run_evaluation(model, audio_dir: str, category: str, writer):
    audio_files = collect_audio_files(audio_dir)
    if len(audio_files) < 2:
        print(f"Need at least 2 audio files in {audio_dir}, found {len(audio_files)}")
        return

    basenames = {f: os.path.basename(f) for f in audio_files}
    pairs = list(itertools.permutations(audio_files, 2))

    param_grid = list(itertools.product(
        GRAIN_SIZES, WINDOW_SIZES, STRIDES, HOPS, AUGMENTATION_CONFIGS,
    ))

    total = len(pairs) * len(param_grid)
    print(f"\n{'=' * 60}")
    print(f"  {category}  ({len(audio_files)} files, {len(pairs)} pairs)")
    print(f"  Grid: {len(GRAIN_SIZES)} grain × {len(WINDOW_SIZES)} win × "
          f"{len(STRIDES)} stride × {len(HOPS)} hop × {len(AUGMENTATION_CONFIGS)} aug")
    print(f"  Total evaluations: {total}")
    print(f"{'=' * 60}\n")

    done = 0
    for grain_size, window_size, stride, hop, (aug, aug_label) in param_grid:
        for target_file, pool_file in pairs:
            done += 1
            t_name = basenames[target_file]
            p_name = basenames[pool_file]
            print(
                f"  [{done}/{total}] g={grain_size} w={window_size} "
                f"s={stride} h={hop} aug={aug_label}  "
                f"{t_name} → {p_name} … ",
                end="", flush=True,
            )
            metrics = run_single(
                model, target_file, pool_file,
                n_q=N_Q,
                window_size=window_size,
                grain_size=grain_size,
                stride=stride,
                hop=hop,
                pool_augmentations=aug,
            )
            print(f"MFCC-L2={metrics['mfcc_l2']:.2f}  FAD={metrics['fad']:.2f}")

            writer.writerow({
                "category": category,
                "target": t_name,
                "pool": p_name,
                "grain_size": grain_size,
                "window_size": window_size,
                "stride": stride,
                "hop": hop,
                "augmentation": aug_label,
                "mfcc_l2": f"{metrics['mfcc_l2']:.4f}",
                "fad": f"{metrics['fad']:.4f}",
            })


def main():
    print("Loading Encodec model …")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz")
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()

        run_evaluation(model, PERCUSSION_DIR, "percussion", writer)
        run_evaluation(model, INSTRUMENTS_DIR, "instruments", writer)

    print(f"\nResults written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
