"""
Generate evaluation figures for the paper.

Reads evaluation_results.csv and writes plots to plots/ directory.
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# ── Style ────────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (7, 4.5),
    "axes.grid": True,
    "grid.alpha": 0.3,
})
PERC_COLOR = "#E07A5F"   # warm terracotta
INST_COLOR = "#3D405B"   # cool slate
PERC_AUG   = "#F2CC8F"   # light gold
INST_AUG   = "#81B29A"   # sage green

OUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
rows = []
with open(os.path.join(os.path.dirname(__file__), "evaluation_results.csv")) as f:
    for r in csv.DictReader(f):
        r["grain_size"]  = int(r["grain_size"])
        r["window_size"] = int(r["window_size"])
        r["stride"]      = int(r["stride"])
        r["hop"]         = int(r["hop"])
        r["mfcc_l2"]     = float(r["mfcc_l2"])
        r["fad"]         = float(r["fad"])
        rows.append(r)

print(f"Loaded {len(rows)} rows")


def subset(cat=None, aug=None, **kwargs):
    out = rows
    if cat is not None:
        out = [r for r in out if r["category"] == cat]
    if aug is not None:
        out = [r for r in out if r["augmentation"] == aug]
    for k, v in kwargs.items():
        out = [r for r in out if r[k] == v]
    return out


def avg(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def metric_by(param, values, cat, aug="none"):
    mfcc = [avg([r["mfcc_l2"] for r in subset(cat=cat, aug=aug, **{param: v})]) for v in values]
    fad  = [avg([r["fad"]     for r in subset(cat=cat, aug=aug, **{param: v})]) for v in values]
    return mfcc, fad


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Effect of Grain Size on FAD & MFCC (perc vs instruments)
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=False)
grains = list(range(1, 6))

for ax, metric, label in [(ax1, "fad", "FAD ↓"), (ax2, "mfcc_l2", "MFCC-L2 ↓")]:
    p_vals = [avg([r[metric] for r in subset(cat="percussion",  aug="none", grain_size=g)]) for g in grains]
    i_vals = [avg([r[metric] for r in subset(cat="instruments", aug="none", grain_size=g)]) for g in grains]
    ax.plot(grains, p_vals, "o-", color=PERC_COLOR, lw=2, label="Percussion")
    ax.plot(grains, i_vals, "s-", color=INST_COLOR, lw=2, label="Instruments")
    ax.set_xlabel("Grain Size")
    ax.set_ylabel(label)
    ax.set_xticks(grains)
    ax.legend()

fig.suptitle("Effect of Grain Size on Resynthesis Quality", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "01_grain_size.png"))
plt.close(fig)
print("  01_grain_size.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Effect of Window Size
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
wins = list(range(1, 6))

for ax, metric, label in [(ax1, "fad", "FAD ↓"), (ax2, "mfcc_l2", "MFCC-L2 ↓")]:
    p_vals = [avg([r[metric] for r in subset(cat="percussion",  aug="none", window_size=w)]) for w in wins]
    i_vals = [avg([r[metric] for r in subset(cat="instruments", aug="none", window_size=w)]) for w in wins]
    ax.plot(wins, p_vals, "o-", color=PERC_COLOR, lw=2, label="Percussion")
    ax.plot(wins, i_vals, "s-", color=INST_COLOR, lw=2, label="Instruments")
    ax.set_xlabel("Window Size")
    ax.set_ylabel(label)
    ax.set_xticks(wins)
    ax.legend()

fig.suptitle("Effect of Window Size on Resynthesis Quality", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "02_window_size.png"))
plt.close(fig)
print("  02_window_size.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Effect of Stride
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
strides = [1, 2, 3]

for ax, metric, label in [(ax1, "fad", "FAD ↓"), (ax2, "mfcc_l2", "MFCC-L2 ↓")]:
    p_vals = [avg([r[metric] for r in subset(cat="percussion",  aug="none", stride=s)]) for s in strides]
    i_vals = [avg([r[metric] for r in subset(cat="instruments", aug="none", stride=s)]) for s in strides]
    ax.plot(strides, p_vals, "o-", color=PERC_COLOR, lw=2, label="Percussion")
    ax.plot(strides, i_vals, "s-", color=INST_COLOR, lw=2, label="Instruments")
    ax.set_xlabel("Stride")
    ax.set_ylabel(label)
    ax.set_xticks(strides)
    ax.legend()

fig.suptitle("Effect of Stride on Resynthesis Quality", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "03_stride.png"))
plt.close(fig)
print("  03_stride.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Effect of Hop
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
hops = [1, 2, 3]

for ax, metric, label in [(ax1, "fad", "FAD ↓"), (ax2, "mfcc_l2", "MFCC-L2 ↓")]:
    p_vals = [avg([r[metric] for r in subset(cat="percussion",  aug="none", hop=h)]) for h in hops]
    i_vals = [avg([r[metric] for r in subset(cat="instruments", aug="none", hop=h)]) for h in hops]
    ax.plot(hops, p_vals, "o-", color=PERC_COLOR, lw=2, label="Percussion")
    ax.plot(hops, i_vals, "s-", color=INST_COLOR, lw=2, label="Instruments")
    ax.set_xlabel("Hop")
    ax.set_ylabel(label)
    ax.set_xticks(hops)
    ax.legend()

fig.suptitle("Effect of Hop on Resynthesis Quality", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "04_hop.png"))
plt.close(fig)
print("  04_hop.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Augmentation: grouped bar chart (FAD & MFCC side by side)
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.5))
cats = ["Percussion", "Instruments"]
x = np.arange(len(cats))
w = 0.3

for ax, metric, label in [(ax1, "fad", "FAD ↓"), (ax2, "mfcc_l2", "MFCC-L2 ↓")]:
    none_vals = [
        avg([r[metric] for r in subset(cat="percussion",  aug="none")]),
        avg([r[metric] for r in subset(cat="instruments", aug="none")]),
    ]
    aug_vals = [
        avg([r[metric] for r in subset(cat="percussion",  aug="augmented")]),
        avg([r[metric] for r in subset(cat="instruments", aug="augmented")]),
    ]
    bars1 = ax.bar(x - w/2, none_vals, w, label="No Augmentation", color=[PERC_COLOR, INST_COLOR], edgecolor="white")
    bars2 = ax.bar(x + w/2, aug_vals,  w, label="Augmented",       color=[PERC_AUG, INST_AUG],     edgecolor="white")
    ax.set_ylabel(label)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.legend()
    # value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)

fig.suptitle("Effect of Pool Augmentation", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "05_augmentation.png"))
plt.close(fig)
print("  05_augmentation.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Grain × Window heatmaps (FAD, no aug, stride=1, hop=1)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

for ax, cat, title in zip(axes, ["percussion", "instruments"], ["Percussion", "Instruments"]):
    mat = np.zeros((5, 5))
    for gi, g in enumerate(range(1, 6)):
        for wi, w in enumerate(range(1, 6)):
            s = subset(cat=cat, aug="none", grain_size=g, window_size=w, stride=1, hop=1)
            mat[gi, wi] = avg([r["fad"] for r in s]) if s else float("nan")
    im = ax.imshow(mat, cmap="YlOrRd_r", aspect="auto", origin="lower")
    ax.set_xticks(range(5))
    ax.set_xticklabels(range(1, 6))
    ax.set_yticks(range(5))
    ax.set_yticklabels(range(1, 6))
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Grain Size")
    ax.set_title(title)
    # annotate cells
    for gi in range(5):
        for wi in range(5):
            val = mat[gi, wi]
            color = "white" if val > (mat.max() + mat.min()) / 2 else "black"
            ax.text(wi, gi, f"{val:.0f}", ha="center", va="center", fontsize=9, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8, label="FAD ↓")

fig.suptitle("FAD by Grain Size × Window Size  (stride=1, hop=1, no aug)", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "06_grain_window_heatmap.png"))
plt.close(fig)
print("  06_grain_window_heatmap.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Grain × Stride interaction (FAD, no aug, win=1, hop=1)
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
grains = list(range(1, 6))
strides = [1, 2, 3]
stride_colors = ["#264653", "#2A9D8F", "#E9C46A"]

for ax, cat, title in [(ax1, "percussion", "Percussion"), (ax2, "instruments", "Instruments")]:
    for si, s in enumerate(strides):
        vals = [avg([r["fad"] for r in subset(cat=cat, aug="none", grain_size=g,
                      window_size=1, stride=s, hop=1)]) for g in grains]
        ax.plot(grains, vals, "o-", color=stride_colors[si], lw=2, label=f"stride={s}")
    ax.set_xlabel("Grain Size")
    ax.set_ylabel("FAD ↓")
    ax.set_title(title)
    ax.set_xticks(grains)
    ax.legend()

fig.suptitle("Grain Size × Stride Interaction  (win=1, hop=1, no aug)", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "07_grain_stride_interaction.png"))
plt.close(fig)
print("  07_grain_stride_interaction.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Overall category comparison (box plot of FAD distributions)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))

perc_fad = [r["fad"] for r in subset(cat="percussion")]
inst_fad = [r["fad"] for r in subset(cat="instruments")]

bp = ax.boxplot(
    [perc_fad, inst_fad],
    labels=["Percussion", "Instruments"],
    patch_artist=True,
    widths=0.5,
    showfliers=False,
    medianprops=dict(color="black", lw=2),
)
bp["boxes"][0].set_facecolor(PERC_COLOR)
bp["boxes"][0].set_alpha(0.7)
bp["boxes"][1].set_facecolor(INST_COLOR)
bp["boxes"][1].set_alpha(0.7)

ax.set_ylabel("FAD ↓")
ax.set_title("FAD Distribution by Category", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "08_category_boxplot.png"))
plt.close(fig)
print("  08_category_boxplot.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Top 10 configurations table as figure
# ══════════════════════════════════════════════════════════════════════════════
combos = defaultdict(list)
for r in rows:
    key = (r["category"], r["grain_size"], r["window_size"], r["stride"], r["hop"], r["augmentation"])
    combos[key].append(r)

ranked = []
for key, rrs in combos.items():
    ranked.append((*key, avg([x["fad"] for x in rrs]), avg([x["mfcc_l2"] for x in rrs]), len(rrs)))
ranked.sort(key=lambda x: x[6])  # sort by FAD

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis("off")
headers = ["Rank", "Category", "Grain", "Window", "Stride", "Hop", "Aug", "FAD", "MFCC-L2"]
table_data = []
for i, (cat, g, w, s, h, aug, fad_v, mfcc_v, n) in enumerate(ranked[:10]):
    table_data.append([
        f"#{i+1}", cat.capitalize(), str(g), str(w), str(s), str(h),
        aug, f"{fad_v:.1f}", f"{mfcc_v:.1f}"
    ])

table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.5)

# colour header
for j in range(len(headers)):
    table[0, j].set_facecolor("#3D405B")
    table[0, j].set_text_props(color="white", fontweight="bold")
# alternate row colours
for i in range(1, 11):
    for j in range(len(headers)):
        table[i, j].set_facecolor("#F4F1DE" if i % 2 == 0 else "white")

ax.set_title("Top 10 Configurations by FAD (lower is better)", fontweight="bold", pad=20)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "09_top_configs_table.png"))
plt.close(fig)
print("  09_top_configs_table.png")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — Radar / parameter sensitivity summary
# ══════════════════════════════════════════════════════════════════════════════
# Show how much each parameter moves FAD (range of means) per category
fig, ax = plt.subplots(figsize=(8, 5))

params = ["grain_size", "window_size", "stride", "hop"]
param_labels = ["Grain Size\n(1→5)", "Window Size\n(1→5)", "Stride\n(1→3)", "Hop\n(1→3)"]
param_vals = {
    "grain_size": range(1, 6),
    "window_size": range(1, 6),
    "stride": range(1, 4),
    "hop": range(1, 4),
}

x_pos = np.arange(len(params))
bar_w = 0.35

perc_ranges = []
inst_ranges = []
for p in params:
    perc_means = [avg([r["fad"] for r in subset(cat="percussion", aug="none", **{p: v})]) for v in param_vals[p]]
    inst_means = [avg([r["fad"] for r in subset(cat="instruments", aug="none", **{p: v})]) for v in param_vals[p]]
    perc_ranges.append(max(perc_means) - min(perc_means))
    inst_ranges.append(max(inst_means) - min(inst_means))

bars1 = ax.bar(x_pos - bar_w/2, perc_ranges, bar_w, label="Percussion", color=PERC_COLOR, edgecolor="white")
bars2 = ax.bar(x_pos + bar_w/2, inst_ranges, bar_w, label="Instruments", color=INST_COLOR, edgecolor="white")

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.annotate(f"{h:.0f}", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)

ax.set_xticks(x_pos)
ax.set_xticklabels(param_labels)
ax.set_ylabel("FAD Range (max − min of parameter means)")
ax.set_title("Parameter Sensitivity: FAD Impact by Category", fontweight="bold")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "10_parameter_sensitivity.png"))
plt.close(fig)
print("  10_parameter_sensitivity.png")

print(f"\nAll plots saved to {OUT_DIR}/")
