"""
dataset_stats.py
----------------
Post-pipeline dataset analysis and visualization tool.

Generates essential statistics and charts about any JSONL dataset file
produced by the VLM fine-tuning pipeline. Designed to run on:
  - train_dataset.jsonl      (Stage 2 raw output)
  - verified_dataset.jsonl   (Stage 3 QA output)
  - final_training_dataset.jsonl (Stage 4 sanitized output)

Usage:
    python support/dataset_stats.py                           # defaults to final_training_dataset.jsonl
    python support/dataset_stats.py --input dataset/train_dataset.jsonl
    python support/dataset_stats.py --input dataset/verified_dataset.jsonl

Output:
    - Console summary table
    - Saved charts in dataset/stats/ directory
    - Optional matplotlib window display

Dependencies:
    pip install matplotlib seaborn
"""

import os
import sys
import json
import re
import argparse
from collections import Counter

# Force UTF-8 output for Windows console emoji support
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ---------------------------------------------------------------------------
# Try imports — give helpful messages if missing
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend by default
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import seaborn as sns
except ImportError:
    print("ERROR: matplotlib and seaborn are required.")
    print("       pip install matplotlib seaborn")
    sys.exit(1)

import numpy as np

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SUPPORT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SUPPORT_DIR)

# Default input — the final pipeline artifact
DEFAULT_INPUT = os.path.join(PROJECT_ROOT, "dataset", "final_training_dataset.jsonl")

# Output directory for charts
STATS_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "stats")

# Label directory — xView2 per-building severity annotations
LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed_labels")

# ---------------------------------------------------------------------------
# Colour palette — consistent dark-mode-friendly scheme
# ---------------------------------------------------------------------------
PALETTE = {
    "bg":        "#1A1A2E",
    "panel":     "#16213E",
    "text":      "#E0E0E0",
    "grid":      "#2A2A4A",
    "accent":    "#3B82F6",
    "bar_colors": [
        "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
        "#EC4899", "#14B8A6", "#F97316", "#6366F1", "#84CC16",
    ],
    # Severity-specific colours — ordered worst→best so heatmap reads intuitively
    "severity_colors": {
        "destroyed":    "#EF4444",   # red
        "major-damage": "#F97316",   # orange
        "minor-damage": "#F59E0B",   # amber
        "no-damage":    "#10B981",   # green
        "un-classified":"#6B7280",   # gray
    },
}

# Canonical severity order (worst → least)
SEVERITY_ORDER = ["destroyed", "major-damage", "minor-damage", "no-damage", "un-classified"]

# Apply global dark style
plt.rcParams.update({
    "figure.facecolor":   PALETTE["bg"],
    "axes.facecolor":     PALETTE["panel"],
    "axes.edgecolor":     PALETTE["grid"],
    "axes.labelcolor":    PALETTE["text"],
    "xtick.color":        PALETTE["text"],
    "ytick.color":        PALETTE["text"],
    "text.color":         PALETTE["text"],
    "grid.color":         PALETTE["grid"],
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     14,
    "axes.titleweight":   "bold",
    "figure.titlesize":   16,
    "figure.titleweight": "bold",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, skipping blank or malformed lines."""
    data = []
    errors = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                errors += 1
                print(f"  ⚠  Line {i}: Invalid JSON — skipped")
    if errors:
        print(f"  Total malformed lines: {errors}")
    return data


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def extract_disaster_event(image_path: str) -> str:
    """Extract disaster event name from image path.
    e.g. 'processed_images/hurricane-michael_00000035_post_disaster.jpg'
          → 'hurricane-michael'
    """
    basename = os.path.basename(image_path)
    # Split from the right to handle compound disaster names (e.g. 'socal-fire')
    parts = basename.rsplit("_", 3)
    return parts[0] if len(parts) >= 4 else basename.split("_")[0]


def extract_disaster_type(event: str) -> str:
    """Map event name to broad disaster category."""
    event_lower = event.lower()
    if "hurricane" in event_lower:
        return "Hurricane"
    elif "fire" in event_lower or "wildfire" in event_lower:
        return "Wildfire"
    elif "flood" in event_lower:
        return "Flood"
    elif "earthquake" in event_lower:
        return "Earthquake"
    elif "volcano" in event_lower:
        return "Volcano"
    elif "tsunami" in event_lower:
        return "Tsunami"
    else:
        return "Other"


def word_count(text: str) -> int:
    return len(text.split())


def has_all_headers(text: str) -> bool:
    return all(h in text for h in ["### 1.", "### 2.", "### 3."])


def has_placeholders(text: str) -> bool:
    return bool(re.search(r"\[.*?\]", text))


def has_first_person(text: str) -> bool:
    patterns = ["I ", "I'm ", "I cannot", "As an AI", "I apologize", "I'm sorry"]
    text_lower = text.lower()
    return any(p.lower() in text_lower for p in patterns)


# ---------------------------------------------------------------------------
# Severity analysis — derived from xView2 label files at runtime
# ---------------------------------------------------------------------------
_SEVERITY_RANK = {"destroyed": 4, "major-damage": 3, "minor-damage": 2,
                  "no-damage": 1, "un-classified": 0}


def load_severity_map() -> dict[str, str]:
    """Build image_basename → max_severity mapping from xView2 label JSONs.

    Reads every JSON file in LABEL_DIR and determines the worst-case building
    severity per image.  Returns a dict keyed by the base name (no extension),
    e.g. ``{"hurricane-harvey_00000158_post_disaster": "major-damage"}``.
    """
    severity_map: dict[str, str] = {}

    if not os.path.isdir(LABEL_DIR):
        print(f"  ⚠  Label directory not found: {LABEL_DIR}")
        print(f"      Severity analysis will be unavailable.")
        return severity_map

    for fname in os.listdir(LABEL_DIR):
        if not fname.endswith(".json"):
            continue
        base = os.path.splitext(fname)[0]
        filepath = os.path.join(LABEL_DIR, fname)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                label_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        buildings = label_data.get("features", {}).get("lng_lat", [])
        max_sev = "un-classified"
        for b in buildings:
            subtype = b.get("properties", {}).get("subtype", "un-classified")
            if _SEVERITY_RANK.get(subtype, 0) > _SEVERITY_RANK.get(max_sev, 0):
                max_sev = subtype
        severity_map[base] = max_sev

    return severity_map


def extract_severity(row: dict, severity_map: dict[str, str]) -> str:
    """Look up max severity for a JSONL row using the preloaded severity map.

    Falls back through:
      1. severity_map (derived from label files)
      2. row['meta']['max_severity'] (future-proof: if the pipeline later embeds it)
      3. row['label']
      4. 'un-classified'
    """
    # Primary: label file lookup
    img = row.get("image", "")
    base = os.path.splitext(os.path.basename(img))[0]
    if base in severity_map:
        return severity_map[base]

    # Fallback: embedded meta (future-proof)
    meta = row.get("meta", {})
    if isinstance(meta, dict) and meta.get("max_severity"):
        return meta["max_severity"]

    # Fallback: legacy label field
    if row.get("label") in SEVERITY_ORDER:
        return row["label"]

    return "un-classified"


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------
def print_report(data: list[dict], input_path: str,
                  severity_map: dict[str, str] | None = None):
    """Print a comprehensive console summary."""
    print("\n" + "=" * 65)
    print("DATASET STATISTICS REPORT")
    print("=" * 65)
    print(f"  Source File  : {input_path}")
    print(f"  Total Samples: {len(data)}")

    if not data:
        print("  ⚠  Dataset is empty. Nothing to report.")
        return

    if severity_map is None:
        severity_map = {}

    # --- Key presence ---
    missing_keys = sum(1 for r in data if not all(k in r for k in ("image", "instruction", "response")))
    if missing_keys:
        print(f"  ⚠  Rows missing required keys: {missing_keys}")

    # --- Word counts ---
    wc = [word_count(r.get("response", "")) for r in data]
    print(f"\n  Response Word Count:")
    print(f"    Min   : {min(wc)}")
    print(f"    Max   : {max(wc)}")
    print(f"    Mean  : {sum(wc)/len(wc):.1f}")
    print(f"    Median: {sorted(wc)[len(wc)//2]}")

    under_80 = sum(1 for w in wc if w < 80)
    over_600 = sum(1 for w in wc if w > 600)
    if under_80:
        print(f"    ⚠  Under 80 words : {under_80} samples")
    if over_600:
        print(f"    ⚠  Over 600 words : {over_600} samples")

    # --- Schema compliance ---
    no_headers = sum(1 for r in data if not has_all_headers(r.get("response", "")))
    placeholders = sum(1 for r in data if has_placeholders(r.get("response", "")))
    first_person = sum(1 for r in data if has_first_person(r.get("response", "")))

    print(f"\n  Quality Checks:")
    print(f"    Missing schema headers : {no_headers} ({no_headers/len(data)*100:.1f}%)")
    print(f"    Placeholder text [...]  : {placeholders} ({placeholders/len(data)*100:.1f}%)")
    print(f"    First-person language   : {first_person} ({first_person/len(data)*100:.1f}%)")

    # --- Disaster distribution ---
    events = [extract_disaster_event(r.get("image", "")) for r in data]
    event_dist = Counter(events)
    types = [extract_disaster_type(e) for e in events]
    type_dist = Counter(types)

    print(f"\n  Disaster Event Distribution:")
    for event, count in event_dist.most_common():
        pct = count / len(data) * 100
        flag = " ⚠ >40%" if pct > 40 else ""
        print(f"    {event.ljust(28)}: {count:4d} ({pct:5.1f}%){flag}")

    print(f"\n  Disaster Type Distribution:")
    for dtype, count in type_dist.most_common():
        pct = count / len(data) * 100
        flag = " ⚠ >40%" if pct > 40 else ""
        print(f"    {dtype.ljust(20)}: {count:4d} ({pct:5.1f}%){flag}")

    # --- Severity distribution ---
    severities = [extract_severity(r, severity_map) for r in data]
    sev_dist = Counter(severities)
    matched = sum(1 for r in data
                  if os.path.splitext(os.path.basename(r.get("image", "")))[0] in severity_map)

    print(f"\n  Damage Severity Distribution:")
    print(f"    (derived from label files for {matched}/{len(data)} images)")
    for sev in SEVERITY_ORDER:
        count = sev_dist.get(sev, 0)
        pct = count / len(data) * 100 if data else 0
        bar = "█" * int(pct / 2)
        flag = " ⚠ underrepresented" if pct < 10 and count > 0 else ""
        flag = " ⚠ dominant class" if pct > 50 else flag
        print(f"    {sev.ljust(18)}: {count:4d} ({pct:5.1f}%)  {bar}{flag}")

    # Imbalance ratio
    present_counts = [sev_dist[s] for s in SEVERITY_ORDER if sev_dist.get(s, 0) > 0]
    if len(present_counts) >= 2:
        ratio = max(present_counts) / min(present_counts)
        flag = " ⚠ severe imbalance" if ratio > 5 else ""
        print(f"\n    Imbalance ratio (max/min): {ratio:.1f}x{flag}")

    # --- 2D cross-tab: disaster type × severity ---
    print(f"\n  2D Balance: Disaster Type × Severity")
    print(f"    {'':20}", end="")
    for sev in SEVERITY_ORDER:
        print(f"  {sev[:8]:>8}", end="")
    print()
    print(f"    {'':-<20}", end="")
    for _ in SEVERITY_ORDER:
        print(f"  {'':->8}", end="")
    print()

    for dtype in sorted(set(types)):
        dtype_rows = [r for r, t in zip(data, types) if t == dtype]
        print(f"    {dtype:<20}", end="")
        for sev in SEVERITY_ORDER:
            count = sum(1 for r in dtype_rows if extract_severity(r, severity_map) == sev)
            cell = str(count) if count > 0 else "·"
            warn = "!" if count > 0 and count < 5 else " "
            print(f"  {warn}{cell:>7}", end="")
        print()
    print(f"\n    ! = fewer than 5 samples (may need augmentation)")

    # --- Perspective distribution (if meta.perspective present — future-proof) ---
    perspectives = [r.get("meta", {}).get("perspective") for r in data if isinstance(r.get("meta"), dict)]
    perspectives = [p for p in perspectives if p]
    if perspectives:
        persp_dist = Counter(perspectives)
        print(f"\n  Augmentation Perspective Distribution:")
        for persp, count in persp_dist.most_common():
            pct = count / len(perspectives) * 100
            print(f"    {persp.ljust(16)}: {count:4d} ({pct:5.1f}%)")

    # --- Instruction diversity ---
    instr_dist = Counter(r.get("instruction", "") for r in data)
    print(f"\n  Instruction Variants: {len(instr_dist)} unique")
    for instr, count in instr_dist.most_common():
        truncated = instr[:70] + "..." if len(instr) > 70 else instr
        print(f"    [{count:4d}] {truncated}")

    # --- Duplicate image paths ---
    img_counts = Counter(r.get("image", "") for r in data)
    duplicates = {k: v for k, v in img_counts.items() if v > 1}
    if duplicates:
        print(f"\n  Duplicate Image Paths (expected with augmentation): {len(duplicates)}")
        for img, count in sorted(duplicates.items(), key=lambda x: -x[1])[:5]:
            print(f"    [{count}x] {img}")
    else:
        print(f"\n  No duplicate image paths")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Chart generation
# ---------------------------------------------------------------------------
def plot_disaster_distribution(data: list[dict], output_dir: str):
    """Bar chart of disaster event distribution."""
    events = [extract_disaster_event(r.get("image", "")) for r in data]
    event_dist = Counter(events)

    labels, values = zip(*event_dist.most_common())
    colors = [PALETTE["bar_colors"][i % len(PALETTE["bar_colors"])] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="none", height=0.6)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Samples")
    ax.set_title("Disaster Event Distribution")
    ax.grid(axis="x", alpha=0.3)

    # Add count labels on bars
    for bar, val in zip(bars, values):
        pct = val / len(data) * 100
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f" {val} ({pct:.1f}%)", va="center", fontsize=10, color=PALETTE["text"])

    # 40% threshold line
    threshold = len(data) * 0.4
    ax.axvline(x=threshold, color="#EF4444", linestyle="--", alpha=0.7, label=f"40% threshold ({int(threshold)})")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "disaster_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_disaster_type_pie(data: list[dict], output_dir: str):
    """Pie chart of broad disaster type categories."""
    events = [extract_disaster_event(r.get("image", "")) for r in data]
    types = [extract_disaster_type(e) for e in events]
    type_dist = Counter(types)

    labels, values = zip(*type_dist.most_common())
    colors = PALETTE["bar_colors"][:len(labels)]

    fig, ax = plt.subplots(figsize=(8, 8))
    pie_chart = ax.pie(
        values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=90,
        textprops={"color": PALETTE["text"], "fontsize": 12},
        pctdistance=0.75,
    )
    for t in pie_chart[2]:
        t.set_fontsize(11)
        t.set_fontweight("bold")

    ax.set_title("Disaster Type Breakdown")
    plt.tight_layout()
    path = os.path.join(output_dir, "disaster_type_pie.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_word_count_histogram(data: list[dict], output_dir: str):
    """Histogram of response word counts."""
    wc = [word_count(r.get("response", "")) for r in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    n, bins, patches = ax.hist(wc, bins=30, color=PALETTE["accent"], edgecolor=PALETTE["panel"], alpha=0.9)

    ax.set_xlabel("Word Count")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Response Word Count Distribution")
    ax.grid(axis="y", alpha=0.3)

    # Mark healthy range
    ax.axvline(x=80, color="#EF4444", linestyle="--", alpha=0.7, label="Min (80 words)")
    ax.axvline(x=600, color="#EF4444", linestyle="--", alpha=0.7, label="Max (600 words)")
    ax.axvline(x=sum(wc)/len(wc), color="#10B981", linestyle="-", alpha=0.8, label=f"Mean ({sum(wc)/len(wc):.0f})")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "word_count_histogram.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_quality_checks(data: list[dict], output_dir: str):
    """Bar chart showing quality check pass/fail rates."""
    checks = {
        "Schema Headers\n(### 1/2/3)": sum(1 for r in data if has_all_headers(r.get("response", ""))),
        "No Placeholders\n([...] free)": sum(1 for r in data if not has_placeholders(r.get("response", ""))),
        "No First-Person\n(I/I'm free)": sum(1 for r in data if not has_first_person(r.get("response", ""))),
        "Word Count\n(80–600)": sum(1 for r in data if 80 <= word_count(r.get("response", "")) <= 600),
        "All Keys\nPresent": sum(1 for r in data if all(k in r for k in ("image", "instruction", "response"))),
    }

    labels = list(checks.keys())
    pass_counts = list(checks.values())
    fail_counts = [len(data) - p for p in pass_counts]
    pass_pcts = [p / len(data) * 100 for p in pass_counts]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(labels))
    width = 0.5

    bars_pass = ax.bar(x, pass_counts, width, color="#10B981", label="Pass", edgecolor="none")
    bars_fail = ax.bar(x, fail_counts, width, bottom=pass_counts, color="#EF4444", label="Fail", edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, ha="center")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Quality Check Results")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars_pass, pass_pcts)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"{pct:.0f}%", ha="center", va="center", fontsize=11, fontweight="bold",
                color="white")

    plt.tight_layout()
    path = os.path.join(output_dir, "quality_checks.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_instruction_diversity(data: list[dict], output_dir: str):
    """Bar chart of instruction variant distribution."""
    instr_dist = Counter(r.get("instruction", "") for r in data)

    if len(instr_dist) <= 1:
        print("  ℹ  Only 1 instruction variant — skipping diversity chart.")
        return

    # Truncate labels for readability
    labels_full = list(instr_dist.keys())
    labels = [l[:50] + "..." if len(l) > 50 else l for l in labels_full]
    values = list(instr_dist.values())
    colors = [PALETTE["bar_colors"][i % len(PALETTE["bar_colors"])] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.6)))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="none", height=0.6)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Samples")
    ax.set_title("Instruction Variant Distribution")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f" {val}", va="center", fontsize=10, color=PALETTE["text"])

    plt.tight_layout()
    path = os.path.join(output_dir, "instruction_diversity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Severity charts
# ---------------------------------------------------------------------------
def plot_severity_distribution(data: list[dict], output_dir: str,
                                severity_map: dict[str, str]):
    """Horizontal bar chart of damage severity distribution with severity-coded colours."""
    severities = [extract_severity(r, severity_map) for r in data]
    sev_dist = Counter(severities)

    # Use canonical order, include only present severities
    labels = [s for s in SEVERITY_ORDER if sev_dist.get(s, 0) > 0]
    values = [sev_dist[s] for s in labels]
    colors = [PALETTE["severity_colors"].get(s, "#6B7280") for s in labels]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor="none", height=0.55)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Samples")
    ax.set_title("Damage Severity Distribution")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, values):
        pct = val / len(data) * 100
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f" {val} ({pct:.1f}%)", va="center", fontsize=11, color=PALETTE["text"])

    # Ideal balance line
    ideal = len(data) / len(labels)
    ax.axvline(x=ideal, color="#8B5CF6", linestyle="--", alpha=0.7,
               label=f"Ideal balance ({ideal:.0f} each)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "severity_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_severity_by_disaster_type(data: list[dict], output_dir: str,
                                    severity_map: dict[str, str]):
    """Grouped bar chart: disaster type on x-axis, severity as grouped bars."""
    events = [extract_disaster_event(r.get("image", "")) for r in data]
    types  = [extract_disaster_type(e) for e in events]
    sevs   = [extract_severity(r, severity_map) for r in data]

    unique_types = sorted(set(types))
    present_sevs = [s for s in SEVERITY_ORDER if any(sv == s for sv in sevs)]

    n_types = len(unique_types)
    n_sevs  = len(present_sevs)
    x = np.arange(n_types)
    width = 0.8 / n_sevs

    fig, ax = plt.subplots(figsize=(max(10, n_types * 2), 6))

    for i, sev in enumerate(present_sevs):
        counts = []
        for dtype in unique_types:
            count = sum(1 for t, s in zip(types, sevs) if t == dtype and s == sev)
            counts.append(count)
        offset = (i - n_sevs / 2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width * 0.9,
                      label=sev,
                      color=PALETTE["severity_colors"].get(sev, "#6B7280"),
                      edgecolor="none")
        # Label non-zero bars
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        str(count), ha="center", va="bottom",
                        fontsize=8, color=PALETTE["text"])

    ax.set_xticks(x)
    ax.set_xticklabels(unique_types, fontsize=11)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Damage Severity by Disaster Type")
    ax.legend(title="Severity", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "severity_by_disaster_type.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_2d_balance_heatmap(data: list[dict], output_dir: str,
                             severity_map: dict[str, str]):
    """
    Heatmap of sample counts: rows = disaster type, cols = severity.
    Cells with < 5 samples are annotated with a warning marker.
    """
    events = [extract_disaster_event(r.get("image", "")) for r in data]
    types  = [extract_disaster_type(e) for e in events]
    sevs   = [extract_severity(r, severity_map) for r in data]

    unique_types = sorted(set(types))
    present_sevs = [s for s in SEVERITY_ORDER if any(sv == s for sv in sevs)]

    # Build count matrix
    matrix = []
    for dtype in unique_types:
        row = []
        for sev in present_sevs:
            count = sum(1 for t, s in zip(types, sevs) if t == dtype and s == sev)
            row.append(count)
        matrix.append(row)

    matrix_np = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, len(present_sevs) * 1.8),
                                    max(4, len(unique_types) * 0.9)))

    # Use a sequential colormap; zero cells stay dark
    im = ax.imshow(matrix_np, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=max(1, matrix_np.max()))

    ax.set_xticks(range(len(present_sevs)))
    ax.set_xticklabels(present_sevs, fontsize=11, rotation=20, ha="right")
    ax.set_yticks(range(len(unique_types)))
    ax.set_yticklabels(unique_types, fontsize=11)
    ax.set_title("2D Balance Heatmap: Disaster Type × Severity\n(! = fewer than 5 samples)")

    # Annotate every cell
    for i in range(len(unique_types)):
        for j in range(len(present_sevs)):
            count = int(matrix_np[i, j])
            label = f"{count}" if count >= 5 else (f"!{count}" if count > 0 else "0")
            color = "black" if matrix_np[i, j] > matrix_np.max() * 0.5 else PALETTE["text"]
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Sample Count", color=PALETTE["text"])
    cbar.ax.yaxis.set_tick_params(color=PALETTE["text"])
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=PALETTE["text"])

    plt.tight_layout()
    path = os.path.join(output_dir, "2d_balance_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_perspective_by_severity(data: list[dict], output_dir: str,
                                  severity_map: dict[str, str]):
    """
    Stacked bar: perspective on x-axis, severity as stacked segments.
    Only rendered if meta.perspective is present in the data.
    """
    rows_with_persp = [r for r in data
                       if isinstance(r.get("meta"), dict) and r["meta"].get("perspective")]
    if not rows_with_persp:
        print("  ℹ  No meta.perspective found — skipping perspective×severity chart.")
        return

    perspectives = [r["meta"]["perspective"] for r in rows_with_persp]
    sevs         = [extract_severity(r, severity_map) for r in rows_with_persp]
    unique_persps = sorted(set(perspectives))
    present_sevs  = [s for s in SEVERITY_ORDER if s in sevs]

    x = np.arange(len(unique_persps))
    bottoms = np.zeros(len(unique_persps))

    fig, ax = plt.subplots(figsize=(10, 5))
    for sev in present_sevs:
        counts = np.array([
            sum(1 for p, s in zip(perspectives, sevs) if p == persp and s == sev)
            for persp in unique_persps
        ], dtype=float)
        ax.bar(x, counts, bottom=bottoms,
               label=sev,
               color=PALETTE["severity_colors"].get(sev, "#6B7280"),
               edgecolor="none")
        bottoms += counts

    ax.set_xticks(x)
    ax.set_xticklabels(unique_persps, fontsize=11)
    ax.set_ylabel("Number of Samples")
    ax.set_title("Augmentation Perspective × Severity\n(checks orthogonal perspectives cover all severity classes)")
    ax.legend(title="Severity", fontsize=9, loc="upper right", framealpha=0.3)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "perspective_by_severity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate statistics and visualizations for a VLM fine-tuning JSONL dataset."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_INPUT,
        help=f"Path to the JSONL dataset file. Default: {DEFAULT_INPUT}"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display charts interactively (requires a GUI backend)."
    )
    args = parser.parse_args()

    input_path = args.input

    # Resolve relative paths
    if not os.path.isabs(input_path):
        input_path = os.path.join(PROJECT_ROOT, input_path)

    if not os.path.exists(input_path):
        # Try fallback files
        fallbacks = [
            os.path.join(PROJECT_ROOT, "dataset", "verified_dataset.jsonl"),
            os.path.join(PROJECT_ROOT, "dataset", "train_dataset.jsonl"),
        ]
        for fb in fallbacks:
            if os.path.exists(fb):
                print(f"  ℹ  {input_path} not found. Falling back to {fb}")
                input_path = fb
                break
        else:
            print(f"  ERROR: No dataset file found at {input_path}")
            print(f"         Also checked: {', '.join(fallbacks)}")
            sys.exit(1)

    # Load data
    print(f"\nLoading: {input_path}")
    data = load_jsonl(input_path)

    if not data:
        print("  ⚠  Dataset is empty. Exiting.")
        sys.exit(1)

    # Load severity map from xView2 label files
    print(f"Loading severity labels from: {LABEL_DIR}")
    severity_map = load_severity_map()
    if severity_map:
        print(f"  Loaded severity data for {len(severity_map)} images")
    else:
        print("  ⚠  No severity data loaded — severity charts will show 'un-classified'")

    # Console report
    print_report(data, input_path, severity_map)

    # Generate charts
    os.makedirs(STATS_OUTPUT_DIR, exist_ok=True)
    print(f"\nGenerating charts → {STATS_OUTPUT_DIR}/")

    # Existing charts
    plot_disaster_distribution(data, STATS_OUTPUT_DIR)
    plot_disaster_type_pie(data, STATS_OUTPUT_DIR)
    plot_word_count_histogram(data, STATS_OUTPUT_DIR)
    plot_quality_checks(data, STATS_OUTPUT_DIR)
    plot_instruction_diversity(data, STATS_OUTPUT_DIR)

    # Severity charts
    plot_severity_distribution(data, STATS_OUTPUT_DIR, severity_map)
    plot_severity_by_disaster_type(data, STATS_OUTPUT_DIR, severity_map)
    plot_2d_balance_heatmap(data, STATS_OUTPUT_DIR, severity_map)
    plot_perspective_by_severity(data, STATS_OUTPUT_DIR, severity_map)

    print(f"\nAll charts saved to: {STATS_OUTPUT_DIR}/")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()


if __name__ == "__main__":
    main()

