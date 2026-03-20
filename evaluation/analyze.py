import csv
from collections import defaultdict

rows = []
with open("evaluation_results.csv") as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["grain_size"] = int(r["grain_size"])
        r["window_size"] = int(r["window_size"])
        r["stride"] = int(r["stride"])
        r["hop"] = int(r["hop"])
        r["mfcc_l2"] = float(r["mfcc_l2"])
        r["fad"] = float(r["fad"])
        rows.append(r)

print(f"Total rows: {len(rows)}")

def avg(vals):
    return sum(vals)/len(vals) if vals else 0

def report(label, subset):
    m = avg([r["mfcc_l2"] for r in subset])
    f = avg([r["fad"] for r in subset])
    print(f"  {label:40s} n={len(subset):5d}  MFCC={m:7.1f}  FAD={f:7.1f}")

# --- By category ---
print("\n=== BY CATEGORY ===")
for cat in ["percussion", "instruments"]:
    sub = [r for r in rows if r["category"] == cat]
    report(cat, sub)

# --- By category x augmentation ---
print("\n=== BY CATEGORY x AUGMENTATION ===")
for cat in ["percussion", "instruments"]:
    for aug in ["none", "augmented"]:
        sub = [r for r in rows if r["category"] == cat and r["augmentation"] == aug]
        report(f"{cat} / {aug}", sub)

# --- By grain_size per category ---
print("\n=== BY GRAIN SIZE (per category, no aug) ===")
for cat in ["percussion", "instruments"]:
    print(f"  [{cat}]")
    for g in range(1, 6):
        sub = [r for r in rows if r["category"] == cat and r["grain_size"] == g and r["augmentation"] == "none"]
        report(f"  grain={g}", sub)

# --- By grain_size per category (augmented) ---
print("\n=== BY GRAIN SIZE (per category, augmented) ===")
for cat in ["percussion", "instruments"]:
    print(f"  [{cat}]")
    for g in range(1, 6):
        sub = [r for r in rows if r["category"] == cat and r["grain_size"] == g and r["augmentation"] == "augmented"]
        report(f"  grain={g}", sub)

# --- By window_size per category ---
print("\n=== BY WINDOW SIZE (per category, no aug) ===")
for cat in ["percussion", "instruments"]:
    print(f"  [{cat}]")
    for w in range(1, 6):
        sub = [r for r in rows if r["category"] == cat and r["window_size"] == w and r["augmentation"] == "none"]
        report(f"  win={w}", sub)

# --- By stride per category ---
print("\n=== BY STRIDE (per category, no aug) ===")
for cat in ["percussion", "instruments"]:
    print(f"  [{cat}]")
    for s in range(1, 4):
        sub = [r for r in rows if r["category"] == cat and r["stride"] == s and r["augmentation"] == "none"]
        report(f"  stride={s}", sub)

# --- By hop per category ---
print("\n=== BY HOP (per category, no aug) ===")
for cat in ["percussion", "instruments"]:
    print(f"  [{cat}]")
    for h in range(1, 4):
        sub = [r for r in rows if r["category"] == cat and r["hop"] == h and r["augmentation"] == "none"]
        report(f"  hop={h}", sub)

# --- Top 5 configs by FAD per category ---
print("\n=== TOP 5 BY FAD (per category) ===")
for cat in ["percussion", "instruments"]:
    print(f"  [{cat}]")
    combos = defaultdict(list)
    for r in rows:
        if r["category"] != cat: continue
        key = (r["grain_size"], r["window_size"], r["stride"], r["hop"], r["augmentation"])
        combos[key].append(r)
    ranked = []
    for key, rrs in combos.items():
        ranked.append((key, avg([x["fad"] for x in rrs]), avg([x["mfcc_l2"] for x in rrs]), len(rrs)))
    ranked.sort(key=lambda x: x[1])
    for i, (key, fad_avg, mfcc_avg, n) in enumerate(ranked[:5]):
        g, w, s, h, a = key
        print(f"  #{i+1} g={g} w={w} s={s} h={h} aug={a:10s} FAD={fad_avg:7.1f} MFCC={mfcc_avg:7.1f} n={n}")

# --- Overall top 5 ---
print("\n=== TOP 5 BY FAD (overall) ===")
combos = defaultdict(list)
for r in rows:
    key = (r["grain_size"], r["window_size"], r["stride"], r["hop"], r["augmentation"])
    combos[key].append(r)
ranked = []
for key, rrs in combos.items():
    ranked.append((key, avg([x["fad"] for x in rrs]), avg([x["mfcc_l2"] for x in rrs]), len(rrs)))
ranked.sort(key=lambda x: x[1])
for i, (key, fad_avg, mfcc_avg, n) in enumerate(ranked[:10]):
    g, w, s, h, a = key
    print(f"  #{i+1} g={g} w={w} s={s} h={h} aug={a:10s} FAD={fad_avg:7.1f} MFCC={mfcc_avg:7.1f} n={n}")

# --- Interaction: grain x window (no aug, stride=1, hop=1) ---
print("\n=== GRAIN x WINDOW (no aug, stride=1, hop=1) ===")
for cat in ["percussion", "instruments"]:
    print(f"  [{cat}]")
    for g in range(1, 6):
        line = f"  grain={g}: "
        for w in range(1, 6):
            sub = [r for r in rows if r["category"] == cat and r["grain_size"] == g
                   and r["window_size"] == w and r["stride"] == 1 and r["hop"] == 1
                   and r["augmentation"] == "none"]
            if sub:
                m = avg([r["mfcc_l2"] for r in sub])
                f_ = avg([r["fad"] for r in sub])
                line += f"w={w}(M={m:.0f},F={f_:.0f})  "
        print(line)

# --- Interaction: grain x stride (no aug, win=1, hop=1) ---
print("\n=== GRAIN x STRIDE (no aug, win=1, hop=1) ===")
for cat in ["percussion", "instruments"]:
    print(f"  [{cat}]")
    for g in range(1, 6):
        line = f"  grain={g}: "
        for s in range(1, 4):
            sub = [r for r in rows if r["category"] == cat and r["grain_size"] == g
                   and r["window_size"] == 1 and r["stride"] == s and r["hop"] == 1
                   and r["augmentation"] == "none"]
            if sub:
                m = avg([r["mfcc_l2"] for r in sub])
                f_ = avg([r["fad"] for r in sub])
                line += f"s={s}(M={m:.0f},F={f_:.0f})  "
        print(line)
