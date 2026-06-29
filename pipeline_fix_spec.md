# Pipeline Fix Spec: `generate_ground_truth.py` & `sanitize_dataset.py`

> **Status: ALL BUGS RESOLVED.** This document is now a historical record of what was fixed and why. The code changes have been applied and verified.

## Agent Instructions

You are being handed two Python scripts that form the data generation pipeline for a VLM fine-tuning project (Qwen2-VL-7B on disaster imagery). Both scripts had confirmed bugs. This document specifies **exactly what was fixed, why, and what the correct output looks like**.

---

## Project Context

- **Goal:** Generate a JSONL training dataset of `(image, instruction, response)` triplets from xBD aerial disaster images, using the Gemini API as the annotation oracle.
- **Dataset target:** 500+ samples, 2D-stratified across `disaster_type × max_severity`.
- **Augmentation strategy:** Rare image+severity combinations get up to 4× prompt-perspective augmentation (Structural / Triage / Logistics / Environmental), each yielding a distinct JSONL row from the same image.
- **Fine-tuning target:** Qwen2-VL-7B via QLoRA on Kaggle T4 GPUs.
- **Downstream consumers of the JSONL:** `sanitize_dataset.py` → `dataset_validator.py` → `dataset_split.py` → fine-tuning script.

---

## File 1: `support/generate_ground_truth.py`

### Bug 1 — `load_existing_progress()` uses wrong deduplication key — ✅ RESOLVED

**Problem:** Keying on `row['image']` alone meant only the first perspective was ever written — reruns skipped all subsequent perspectives for any image that already had one row.

**Fix applied:** Deduplication now uses `f"{row['image']}::{perspective}"` as the composite key. Legacy rows (without a `meta` field) are keyed as `image::default`.

---

### Bug 2 — Disaster emphasis leaked into saved instruction — ✅ RESOLVED

**Problem:** The spec originally proposed saving `instruction = AUGMENTATION_PROMPTS[perspective] + get_disaster_emphasis(disaster_type)`. This created contradictory instructions (e.g., "Focus exclusively on logistics... Focus on collapse mode...").

**Post-review correction:** The disaster emphasis guides Gemini's *generation quality* via `prompt_text` but is **NOT** saved into the JSONL `instruction` field:

```python
# What Gemini receives (for generation quality):
prompt_text = (
    f"{AUGMENTATION_PROMPTS[perspective]}"
    f"{get_disaster_emphasis(disaster_type)}\n\n"
    f"Metadata Annotations:\n{metadata_summary}"
)

# What gets saved to JSONL (clean, non-contradictory):
instruction = AUGMENTATION_PROMPTS[perspective]  # perspective only
```

---

### Bug 3 — JSONL schema missing `meta` field — ✅ RESOLVED

**Fix applied:** Every new row now includes:
```json
{
  "image": "processed_images/hurricane-harvey_00000042_post_disaster.jpg",
  "instruction": "<perspective-only instruction>",
  "response": "<Gemini-generated response>",
  "meta": {
    "disaster_type": "hurricane",
    "max_severity": "destroyed",
    "perspective": "structural",
    "multiplier_applied": true,
    "model": "gemini-3.5-flash"
  }
}
```

> **Note on legacy rows:** The 220 existing rows lack a `meta` field. Their metadata can be backfilled from `processed_labels/*.json` using `parse_xbd_json()` to extract `disaster_type` and `max_severity`. The `perspective` for legacy rows is `"default"` (single generic prompt).

---

### Tiered multiplier logic — ✅ IMPLEMENTED

```python
def get_multiplier(available: int, needed: int) -> int:
    ratio = needed / max(available, 1)
    if ratio <= 1: return 1   # surplus — Structural only
    elif ratio <= 2: return 2 # mild deficit — Structural + Triage
    elif ratio <= 3: return 3 # moderate deficit — + Logistics
    else: return 4            # severe deficit — all 4 perspectives
```

Perspectives applied in fixed priority: `["structural", "triage", "logistics", "environmental"]`.

---

### `max_severity` helper — ✅ IMPLEMENTED

```python
SEVERITY_ORDER = ['destroyed', 'major-damage', 'minor-damage', 'no-damage', 'un-classified']

def get_max_severity(features: list) -> str:
    present = {f['properties'].get('subtype', 'un-classified') for f in features}
    for level in SEVERITY_ORDER:
        if level in present:
            return level
    return 'un-classified'
```

`un-classified`-only images are excluded from the 2D stratification quota.

---

### Quota math — ✅ IMPLEMENTED

Uses `math.ceil` per cell. The global `TARGET_SAMPLE_COUNT` acts as the hard ceiling regardless of cell quotas summing over.

---

## File 2: `support/sanitize_dataset.py`

### Bug 4 — Round-robin instruction overwrite clobbers augmentation instructions — ✅ RESOLVED

**Fix applied:** Conditional replacement — only overwrites the instruction if it matches the old hardcoded generic string exactly:

```python
OLD_GENERIC = "Analyze this aerial view and identify priority zones for search and rescue operations."

if row.get('instruction', '').strip() == OLD_GENERIC:
    row['instruction'] = INSTRUCTION_VARIANTS[len(clean_data) % len(INSTRUCTION_VARIANTS)]
# else: keep the correctly generated perspective-specific instruction
```

---

### Bug 5 — Disaster type tracking via filename split is fragile — ✅ RESOLVED

**Fix applied:** Prefers `meta.disaster_type`, falls back to filename parse for legacy rows:

```python
disaster_type = row.get('meta', {}).get('disaster_type') or os.path.basename(row['image']).split('_')[0]
disaster_counts[disaster_type] += 1
```

---

## Expected final JSONL format

Every row in `final_training_dataset.jsonl` must satisfy:

```python
assert all(k in row for k in ('image', 'instruction', 'response'))
assert 80 <= len(row['response'].split()) <= 600
assert not re.search(r'\[.*?\]', row['response'])           # no placeholders
assert not any(p in row['response'].lower() for p in [      # no first-person
    "i ", "i'm ", "i cannot", "as an ai", "i apologize", "i'm sorry"
])
```

Rows are shuffled (`random.seed(42)`) before writing. The `meta` field is preserved if present.

---

## Definition of Done — ✅ ALL COMPLETE

- [x] `load_existing_progress()` uses `image::perspective` composite key
- [x] Saved instruction = `AUGMENTATION_PROMPTS[perspective]` only (disaster emphasis goes to `prompt_text` only)
- [x] Every new JSONL row has a `meta` field with `disaster_type`, `max_severity`, `perspective`, `multiplier_applied`, `model`
- [x] Tiered multiplier (1/2/3/4×) used instead of flat 4× for all deficit buckets
- [x] `max_severity` uses `SEVERITY_ORDER` list, not `max()` or set operations
- [x] `un-classified`-only images excluded from 2D stratification
- [x] `sanitize_dataset.py` only replaces instruction when it matches the old generic string exactly
- [x] `sanitize_dataset.py` uses `meta.disaster_type` for disaster type tracking with filename fallback
