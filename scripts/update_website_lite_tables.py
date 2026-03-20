from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "evaluation" / "evaluation_results.csv"
INDEX_PATH = ROOT / "website-lite" / "index.html"

BEGIN = "<!-- BEGIN EVAL TABLES -->"
END = "<!-- END EVAL TABLES -->"


def _mean(items: Iterable[float]) -> float:
    items = list(items)
    if not items:
        raise ValueError("mean of empty list")
    return sum(items) / len(items)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _group_mean(
    rows: list[dict[str, str]],
    group_keys: tuple[str, ...],
    value_key: str,
) -> dict[tuple[str, ...], float]:
    buckets: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        buckets[key].append(float(r[value_key]))
    return {k: _mean(v) for k, v in buckets.items()}


def _sorted_unique(rows: list[dict[str, str]], key: str) -> list[str]:
    try:
        return [str(v) for v in sorted({int(r[key]) for r in rows})]
    except Exception:
        return sorted({r[key] for r in rows})


def _f1(x: float) -> str:
    return f"{x:.1f}"


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    thead = "".join(f"<th>{h}</th>" for h in headers)
    body = "\n".join(
        "          <tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
    )
    return (
        '      <table class="tbl">\n'
        "        <thead>\n"
        f"          <tr>{thead}</tr>\n"
        "        </thead>\n"
        "        <tbody>\n"
        f"{body}\n"
        "        </tbody>\n"
        "      </table>\n"
    )


def _generate_block(rows: list[dict[str, str]]) -> str:
    cats = ["percussion", "instruments"]
    aug_labels = _sorted_unique(rows, "augmentation")

    mean_cat_fad = _group_mean(rows, ("category",), "fad")
    mean_cat_mfcc = _group_mean(rows, ("category",), "mfcc_l2")

    grains = _sorted_unique(rows, "grain_size")
    mean_grain_fad = _group_mean(rows, ("category", "grain_size"), "fad")
    mean_grain_mfcc = _group_mean(rows, ("category", "grain_size"), "mfcc_l2")

    strides = _sorted_unique(rows, "stride")
    mean_stride_fad = _group_mean(rows, ("category", "stride"), "fad")

    windows = _sorted_unique(rows, "window_size")
    mean_window_fad = _group_mean(rows, ("category", "window_size"), "fad")

    hops = _sorted_unique(rows, "hop")
    mean_hop_fad = _group_mean(rows, ("category", "hop"), "fad")

    mean_aug_fad = _group_mean(rows, ("category", "augmentation"), "fad")
    mean_aug_mfcc = _group_mean(rows, ("category", "augmentation"), "mfcc_l2")

    out: list[str] = []
    out.append('      <p class="muted"><em>Auto-generated from <code>evaluation/evaluation_results.csv</code>. '
               'Run <code>python scripts/update_website_lite_tables.py</code> to refresh.</em></p>')

    out.append("      <h3>Category means</h3>")
    out.append(
        _html_table(
            ["Category", "Mean FAD", "Mean MFCC‑L2"],
            [
                ["Percussion", _f1(mean_cat_fad[("percussion",)]), _f1(mean_cat_mfcc[("percussion",)])],
                ["Instruments", _f1(mean_cat_fad[("instruments",)]), _f1(mean_cat_mfcc[("instruments",)])],
            ],
        ).rstrip()
    )

    out.append("      <h3>Grain size</h3>")
    out.append(
        _html_table(
            ["Grain", "Perc FAD", "Perc MFCC‑L2", "Inst FAD", "Inst MFCC‑L2"],
            [
                [
                    g,
                    _f1(mean_grain_fad[("percussion", g)]),
                    _f1(mean_grain_mfcc[("percussion", g)]),
                    _f1(mean_grain_fad[("instruments", g)]),
                    _f1(mean_grain_mfcc[("instruments", g)]),
                ]
                for g in grains
            ],
        ).rstrip()
    )

    out.append("      <h3>Stride</h3>")
    out.append(
        _html_table(
            ["Stride", "Perc FAD", "Inst FAD"],
            [
                [s, _f1(mean_stride_fad[("percussion", s)]), _f1(mean_stride_fad[("instruments", s)])]
                for s in strides
            ],
        ).rstrip()
    )

    out.append("      <h3>Window size</h3>")
    out.append(
        _html_table(
            ["Window", "Perc FAD", "Inst FAD"],
            [
                [w, _f1(mean_window_fad[("percussion", w)]), _f1(mean_window_fad[("instruments", w)])]
                for w in windows
            ],
        ).rstrip()
    )

    out.append("      <h3>Hop</h3>")
    out.append(
        _html_table(
            ["Hop", "Perc FAD", "Inst FAD"],
            [[h, _f1(mean_hop_fad[("percussion", h)]), _f1(mean_hop_fad[("instruments", h)])] for h in hops],
        ).rstrip()
    )

    out.append("      <h3>Augmentation</h3>")
    out.append(
        _html_table(
            ["Category", "Aug", "FAD", "MFCC‑L2"],
            [
                [c, a, _f1(mean_aug_fad[(c, a)]), _f1(mean_aug_mfcc[(c, a)])]
                for c in cats
                for a in aug_labels
            ],
        ).rstrip()
    )

    return "\n".join(out).rstrip() + "\n"


def _replace_between(text: str, begin: str, end: str, replacement: str) -> str:
    b = text.find(begin)
    e = text.find(end)
    if b == -1 or e == -1 or e < b:
        raise SystemExit(f"Could not find marker block in {INDEX_PATH} ({begin}..{end}).")
    head = text[: b + len(begin)]
    tail = text[e:]
    return head + "\n\n" + replacement + "\n" + tail


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing {CSV_PATH}")
    if not INDEX_PATH.exists():
        raise SystemExit(f"Missing {INDEX_PATH}")

    rows = _read_rows(CSV_PATH)
    block = _generate_block(rows)
    html = INDEX_PATH.read_text()
    updated = _replace_between(html, BEGIN, END, block)
    INDEX_PATH.write_text(updated)
    print(f"Updated eval tables in {INDEX_PATH}")


if __name__ == "__main__":
    main()

