# DATASET QUALITY SPECIFICATION
## Synthetic VLM Disaster Rescue Dataset — Engineering Checklist

**Document Type:** Agent Delegation Spec  
**Pipeline Scope:** Stage 1 (Image Standardizer) → Stage 2 (Ground Truth Generation) → Stage 3 (QA Validator) → Stage 4 (Sanitizer & Instruction Augmentation) → Stage 5 (Train/Val/Test Split)  
**Downstream Consumer:** Qwen2-VL-7B, Qwen2-VL-2B, LLaVA-1.5-7B, InternVL2-8B, PaliGemma-3B fine-tuning via QLoRA  
**Non-Negotiable Contract:** Every accepted sample must conform to `{"image": str, "instruction": str, "response": str, "meta": dict}`. The `meta` block is an optional nested dictionary for dataset pipeline tracking; no other nested objects are permitted.

**Source Dataset:** xView2 Challenge Dataset (xBD) — 2,799 labelled post-disaster train images + 1,866 **unlabelled** test images (test set is unusable for this pipeline).

---

## CRITICAL CONTEXT FOR AGENTS

You are building a fine-tuning dataset for Vision Language Models (VLMs) that must generalize across **multiple VLM architectures** and **multiple disaster types**. A dataset that looks complete but violates any rule in this document will produce a model that either overfits to one prompt style, hallucinates rescue guidance, or fails entirely on unseen disaster categories. Each rule below exists because it directly maps to a known failure mode in VLM fine-tuning.

### Current Workspace Layout

```text
CSE499AB_project/
├── data/
│   ├── xView2 Challenge Dataset - train and test/   # Raw source (75% train / 25% test)
│   │   ├── train/
│   │   │   ├── images/          # 5,598 PNGs (2,799 pre + 2,799 post)
│   │   │   └── labels/          # 2,799 JSON label files (post-disaster only)
│   │   ├── test/
│   │   │   └── images/          # 1,866 PNGs — NO LABELS (unusable for our pipeline)
│   │   └── dataset metadata.json
│   ├── processed_images/        # Stage 1 output: standardized JPEGs
│   ├── processed_labels/        # Stage 1 output: copied JSON metadata
│   └── processed_tracker.txt    # Stage 1 state file
│
├── dataset/
│   ├── train_dataset.jsonl      # Stage 2 output: raw generated samples
│   ├── verified_dataset.jsonl   # Stage 3 output: human-approved samples (INTERMEDIATE)
│   └── final_training_dataset.jsonl  # Stage 4 output: sanitized, shuffled, ready for split
│
├── support/
│   ├── image_standardizer.py    # Stage 1 script
│   ├── generate_ground_truth.py # Stage 2 script
│   ├── dataset_validator.py     # Stage 3 script (customtkinter GUI)
│   ├── sanitize_dataset.py      # Stage 4 script
│   └── dataset_stats.py         # Post-pipeline stats & visualization
│
├── requirements.txt
├── .env                         # GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
└── README.md
```

### xView2 Source Dataset — Disaster Type Inventory

The xView2 train split contains **2,799 post-disaster images** across **10 disaster events**:

| Disaster Event | Post-Disaster Images | Type Category |
|---|---|---|
| socal-fire | 823 | Wildfire |
| hurricane-michael | 343 | Hurricane |
| hurricane-florence | 319 | Hurricane |
| hurricane-harvey | 319 | Hurricane |
| midwest-flooding | 279 | Flood |
| hurricane-matthew | 238 | Hurricane |
| santa-rosa-wildfire | 226 | Wildfire |
| mexico-earthquake | 121 | Earthquake |
| palu-tsunami | 113 | Tsunami |
| guatemala-volcano | 18 | Volcano |

> [!WARNING]
> The current 500-image processed batch covers only **5 of 10 disaster events** (all 4 hurricanes + guatemala-volcano). Earthquake, wildfire, flood, and tsunami events are **completely absent** from the generated dataset. Future batches must deliberately target these under-represented categories.

---

## STAGE 1 — IMAGE STANDARDIZER

**Script:** `support/image_standardizer.py`  
**Input:** `data/xView2 Challenge Dataset - train and test/train/images/*.png`  
**Output:** `data/processed_images/*.jpg` + `data/processed_labels/*.json` + `data/processed_tracker.txt`

### 1.1 Resolution and Format

- **Longest edge must be exactly 1280px, aspect ratio preserved.** Do not resize to a fixed square (e.g., 512×512). Squashing a wide satellite image destroys the spatial proportions rescuers depend on. Different VLMs handle aspect ratio differently — Qwen2-VL tiles dynamically, LLaVA-1.5 expects 336px square patches, InternVL2 uses 448px tiles — but all of them start from the raw image you provide. If your raw image is distorted, every model downstream inherits that distortion.
- **Output format must be JPEG, not PNG.** Lossless PNGs of satellite imagery can exceed 20MB per image. On a Kaggle T4 with 15GB VRAM, loading a batch of 4 PNG images into the vision encoder's embedding space can trigger an OOM before a single gradient is computed.
- **JPEG quality setting: 95.** This preserves fine structural detail in rubble regions while keeping file sizes manageable. Below 85, compression artifacts appear as false texture features and can confuse the vision encoder's edge detectors.

> [!NOTE]
> **Current state:** All 500 processed images were saved at quality 95 via `image_standardizer.py`. This is the established baseline for this project.

### 1.2 Color Space Enforcement

- **Convert all images to RGB mode before saving.** The xView2 source images are RGB PNGs, but the standardizer applies `image.convert('RGB')` unconditionally as a safety net. This correctly handles any RGBA images (e.g., Copernicus exports with transparency channels) by dropping the alpha channel. All five target VLMs expect 3-channel RGB input — an RGBA image will produce incorrect embeddings or crash the data collator.
- **Grayscale handling:** The xView2 dataset does not contain grayscale images. If future dataset expansion introduces sources with single-channel imagery (e.g., thermal or SAR exports), those images should be **logged and skipped**, not converted to pseudo-RGB by repeating the channel — this creates a statistically different input distribution that confuses colour-calibrated feature maps.

### 1.3 Integrity Validation

- **Validate every output image can be re-opened after saving.** Write the file, then immediately attempt `Image.open(path).verify()`. A corrupted JPEG that silently writes without error will crash the training data loader mid-epoch, not at startup.
- **Minimum resolution floor: 256×256px after resize.** If a source image is smaller than this after standardisation, it contains insufficient spatial information for a VLM to identify structural features. Log and skip it.
- **MD5 deduplication (recommended for future expansion):** For a single-operator pipeline on xView2, the filename-based tracker is sufficient since xView2 filenames are globally unique. For multi-source datasets (FloodNet, AIDER, RescueNet), compute and store MD5 hashes to catch cross-dataset duplicates where different filenames may refer to the same image.

### 1.4 Tracker File

**Current format** (`processed_tracker.txt`): One base filename per line (e.g., `socal-fire_00001184_post_disaster`). This enables resumable batch processing — the script checks this file before processing any image.

**Recommended upgrade for multi-team use:** `original_filename | output_filename | md5_hash | timestamp | status (success/skipped/failed)`. This richer format enables debugging pipeline failures across team members working on different datasets.

---

## STAGE 2 — GROUND TRUTH GENERATION

**Script:** `support/generate_ground_truth.py`  
**Input:** `data/processed_images/*.jpg` + `data/processed_labels/*.json`  
**Output:** `dataset/train_dataset.jsonl`  
**API:** Google Gemini (gemini-3.5-flash primary, gemini-2.5-flash fallback) via `google-genai` SDK

### 2.1 System Prompt Engineering

- **The system prompt must explicitly specify the 3-part schema with section headers.** Do not assume the generative model will infer the schema from examples alone. Every API call must include the following instruction verbatim or equivalent:

```
You are an expert disaster response and structural engineering analyst.
You will be provided with an aerial post-disaster image AND localized
metadata annotations. Your task is to combine the visual evidence from the
image with the hard numbers from the metadata to generate a highly
professional, concise, and tactical rescue report.

You must strictly output your assessment following this schema without deviation:
### 1. Priority Zones (Geospatial Mapping)
### 2. Structural Damage & Collapse Characterization
### 3. Hazard Avoidance & Logistics Constraints

Constraint: Do not include introductory or concluding pleasantries.
Maintain an authoritative, objective, and operational tone.
```

- **Set API temperature to 0.3 or lower.** Higher temperatures produce creative, varied language — which is the opposite of what you want for a schema-constrained rescue report. Consistency in structure is more important than variety in vocabulary at generation time. Instruction diversity is handled separately by the sanitizer (Stage 4).

> [!NOTE]
> **Implementation Status:** `generate_ground_truth.py` correctly sets `temperature=0.3` and `max_output_tokens=600`.

### 2.2 Multimodal Payload Requirements

- **Always pass both the image AND the parsed metadata text.** Never pass only the image. The visual encoder alone cannot reliably count individual buildings from satellite altitude — it estimates. The hard numbers from the JSON labels (e.g., "10 intact, 3 destroyed, 1 major damage") ground the response in factual accuracy. The image handles what the JSON cannot: visual hazards like blocked roads, mudflows, debris fields, or proximity of collapse to water.
- **The metadata string passed to the API must follow a fixed template:**

```
Disaster Type: {disaster_type}
Total Buildings Detected: {total_buildings}
Damage Assessment: {destroyed} destroyed, {major_damage} major damage,
{minor_damage} minor damage, {intact} intact.
```

> [!NOTE]
> For future multi-source datasets lacking JSON metadata (e.g., AIDER), state this explicitly in the prompt: `Metadata: None available. Disaster Category: {category}. Rely entirely on visual analysis.` This makes the model's reliance on visual reasoning explicit rather than implicit.

### 2.3 Response Quality Gates

Quality gates for this pipeline are divided between **Stage 2** (basic structural validation at generation time) and **Stage 4** (the sanitizer, which handles deep cleaning post-QA).

**Stage 2 should enforce (before appending to `train_dataset.jsonl`):**
- **Schema header check:** All three headers (`### 1.`, `### 2.`, `### 3.`) must be present. Reject and retry if any is missing.
- **Minimum word count:** Response body must contain ≥ 120 words. Reject if below.
- **No first-person language:** Reject responses containing `"I "`, `"I'm "`, `"I cannot"`, `"As an AI"`.

> [!NOTE]
> **Current state:** `generate_ground_truth.py` implements these pre-QA checks natively via the `passes_qa_gates()` function.

**Stage 4 enforces (post-QA, on `verified_dataset.jsonl`):**
- Placeholder regex: `\[.*?\]` — catches `[insert coordinates]`, `[TBD]`, etc.
- Missing key protection (malformed JSON rows)
- Apology/meta-response language filtering
- Maximum word count enforcement (≤ 500 words)

### 2.4 Disaster-Type-Specific Prompt Tuning

> [!NOTE]
> **Status: Implemented.** `generate_ground_truth.py` uses `DISASTER_PROMPTS` mapped by detected disaster type, injected into `prompt_text` alongside `AUGMENTATION_PROMPTS`.

| Disaster Type | Prompt Emphasis Addition |
|---|---|
| Earthquake / Building Collapse | "Focus on collapse mode (pancake, lean-over, V-space), void identification, and column/stairwell proximity." |
| Flood / Hurricane | "Focus on water ingress depth estimation, roof refuge identification, and waterborne access routes." |
| Volcano / Lahar | "Focus on flow path direction, isolation risk, and secondary hazard timeline (re-flow, gas)." |
| Wildfire | "Focus on burn perimeter, structure integrity after thermal stress, and ember-cast secondary ignition zones." |
| Tsunami | "Focus on wave direction indicators, debris field extent, and elevated-structure refuge viability." |

### 2.5 Coverage and Balance Requirements

- **Target distribution across disaster types: no single type exceeding 40% of the final dataset.** Track running counts during generation. If a category exceeds 40% before others are complete, pause generation on that category and redirect effort.
- **Include "low damage" and "no damage" scenes.** A dataset containing only catastrophic destruction trains a model to always output high-urgency responses. Real rescuers need the model to correctly identify low-priority zones too. Target at least 15% of samples being intact-structure or minimal-damage scenes with appropriately de-escalated responses.
- **Include scenes where the response is a clear negative priority signal.** Example response opening for an intact-structure scene: `"No immediate structural collapse or survivor entrapment risk is identified in this sector. Priority for ground teams is low..."` The model must learn this output pattern as confidently as it learns high-urgency patterns.

> [!WARNING]
> **Current state:** The 220 generated samples break down as: hurricane-michael 27.7%, hurricane-harvey 25.9%, hurricane-florence 22.7%, hurricane-matthew 20.0%, guatemala-volcano 3.6%. All four hurricane events dominate (96.4%). Five disaster types with 1,562 source images (socal-fire, midwest-flooding, mexico-earthquake, palu-tsunami, santa-rosa-wildfire) have **zero representation**. Future generation runs must use `disaster_filter` in `image_standardizer.py` to target these gaps.

### 2.6 Image Path Format

- **The `image` field must store the relative path from the `data/` directory.** Correct: `"processed_images/hurricane-michael_00000035_post_disaster.jpg"`. Incorrect: `"/home/user/project/data/processed_images/..."` or `"C:\\Users\\..."`. Absolute paths break the dataset the moment it is uploaded to Kaggle or shared between team members.

> [!NOTE]
> The current `generate_ground_truth.py` correctly writes paths as `processed_images/{base_name}.jpg` (relative to `data/`). The QA validator and sanitizer must resolve these paths against the `data/` directory, not the project root.

---

## STAGE 3 — QA VALIDATOR

**Script:** `support/dataset_validator.py`  
**Input:** `dataset/train_dataset.jsonl`  
**Output:** `dataset/verified_dataset.jsonl` (INTERMEDIATE — not the final training file)

### 3.1 Annotator Calibration Before QA Begins

- **All QA annotators should pass a calibration round before reviewing real data.** Create 10 gold-standard samples — 5 correct, 3 with known placeholder errors, 2 with schema violations — and have every annotator review them blind. Only annotators who correctly flag all 8 problematic samples should proceed to QA real data.

> [!NOTE]
> **Status: Not yet implemented.** No calibration sample set currently exists.

### 3.2 Acceptance Criteria

A sample should only be accepted if it meets ALL of the following:

1. **All three section headers are present and correctly formatted** (exact markdown match).
2. **Geographic references are specific.** Acceptable: "northeast quadrant", "upper-left cluster", "central building complex". Unacceptable: "some buildings", "an area", "certain structures".
3. **Priority guidance is actionable for a non-expert.** A person who is not a structural engineer must be able to read the Priority Zones section and know where to send a team.
4. **No hazardous misguidance.** Any response that directs rescuers toward a visually unstable structure without flagging the risk must be edited or rejected.
5. **No overconfident void prediction.** Responses claiming precise survivor location (e.g., "survivor is on Floor 3, Room 4") must be edited to probability language ("most probable void space is at the column intersection on the northern face").
6. **Length is appropriate.** Reject responses under 80 words or over 600 words.

### 3.3 Edit Logging

> [!WARNING]
> **Status: Not implemented.** The current `dataset_validator.py` saves edited text directly but does **not** log diffs, editor IDs, timestamps, or edit reasons. This means systematic generation failures cannot be diagnosed from QA data alone.

**Recommended implementation:** Every manual edit should log: `original_text | edited_text | editor_id | timestamp | edit_reason (dropdown: placeholder_removed | specificity_added | hazard_corrected | schema_fixed | other)`. If 30% of samples require the same type of edit, the Stage 2 prompt needs fixing — not just the samples.

### 3.4 Rejection Rate Monitoring

- **A rejection rate below 5% is a red flag, not a success.** It indicates the QA annotator is rubber-stamping samples rather than reviewing them. Investigate immediately.
- **A rejection rate above 40% indicates Stage 2 is broken.** Stop QA, fix the generation prompt, and regenerate the batch. Do not continue QA-ing a fundamentally broken generation output.
- **Target rejection rate: 10–25%.** This range indicates the generator is working but human oversight is catching real errors.

### 3.5 Post-QA Balance Check

Before closing a QA session, run a distribution check on `verified_dataset.jsonl`:

```python
from collections import Counter
import json

with open("dataset/verified_dataset.jsonl") as f:
    data = [json.loads(l) for l in f if l.strip()]

dist = Counter(
    line['image'].split('/')[1].rsplit('_', 3)[0]   # extracts disaster event
    for line in data
)
for k, v in dist.most_common():
    print(f"{k}: {v} ({v/len(data)*100:.1f}%)")
```

If any single disaster type exceeds 40%, flag it before submitting to the sanitizer.

---

## STAGE 4 — SANITIZER & INSTRUCTION AUGMENTATION

**Script:** `support/sanitize_dataset.py` (TODO — see reference implementation below)  
**Input:** `dataset/verified_dataset.jsonl`  
**Output:** `dataset/final_training_dataset.jsonl`

This stage transforms the human-approved intermediate dataset into a training-ready file. It performs three functions: **cleaning**, **instruction diversification**, and **shuffling**.

### 4.1 Cleaning Rules

| Check | Action on Failure |
|---|---|
| Missing keys (`image`, `instruction`, `response`) | Skip row, log warning |
| Placeholder text (regex: `\[.*?\]`) | Skip row, log as dirty |
| First-person language (`"I "`, `"I'm "`, `"I cannot"`, `"As an AI"`) | Skip row, log as dirty |
| Apology language (`"I apologize"`, `"I'm sorry"`, `"unfortunately I"`) | Skip row, log as dirty |
| Invalid JSON line | Skip row, log warning |
| Response word count < 80 or > 600 | Skip row, log as out-of-range |

### 4.2 Instruction Diversification

The raw dataset (legacy data) used a single instruction string for all samples. To prevent the model from overfitting to one prompt phrasing, the sanitizer identifies legacy rows (via the hardcoded string) and replaces the instruction with a **round-robin rotation** across these variants.

*Note: Newly generated Smart Augmentation rows inherently have diverse, perspective-specific instructions and are bypassed by the round-robin replacer.*

```python
INSTRUCTION_VARIANTS = [
    "Analyze this aerial view and identify priority zones for search and rescue operations.",
    "What are the key rescue priorities visible in this post-disaster aerial image?",
    "Assess this disaster scene for survivor localization and structural hazards.",
    "Identify structural failures and safe extraction routes from this satellite image.",
    "Provide a tactical rescue operations briefing based on this aerial view.",
    "Evaluate the structural damage and highlight logistics constraints in this area."
]
```

Round-robin (`INSTRUCTION_VARIANTS[index % len(INSTRUCTION_VARIANTS)]`) guarantees balanced distribution across variants, unlike `random.choice()` which can produce significant imbalance on small datasets.

### 4.3 Reproducibility & Shuffle

- **`random.seed(42)` must be set** before any randomized operation. Two runs of the sanitizer must produce byte-identical output.
- **Shuffle the output before writing.** Without shuffling, all Guatemala volcano samples appear first, then all Hurricane Michael samples, etc. During training, the model sees one disaster type for entire epochs before switching — producing biased gradient updates.

### 4.4 Output Report

The sanitizer must print a summary report showing:
- Total input samples, dirty samples removed, final clean count
- Disaster type distribution (count and percentage) of the clean output
- Output file path

### 4.5 Reference Implementation

See `claude sonnet 4.6 web reponse for sanitization.txt` in the project root for a reviewed drop-in implementation incorporating all rules above.

---

## CROSS-CUTTING REQUIREMENTS (Apply to All Stages)

### Train / Validation / Test Split

> [!IMPORTANT]
> The xView2 dataset ships with a **75/25 train/test split at the source image level**. The test set (1,866 images) has **no labels** and is **unusable** for our pipeline. Our entire pipeline operates exclusively on the 2,799 labelled post-disaster images from the `train/` split.

The train/val/test split for VLM fine-tuning is performed on the **final JSONL file** (`final_training_dataset.jsonl`), not on the source images. This is a distinct operation from xView2's original split.

- **Split must be stratified by both disaster type AND disaster event (geographic event), not just by type.** If Hurricane Michael has 50 samples, the split must include Hurricane Michael images in train, val, and test — not 50 train, 0 val, 0 test. Similarly, the same geographic tile or building cluster must never appear in both train and test splits. Use event-level grouping before random splitting.
- **Final split ratio: 80% train / 10% val / 10% test.** The smaller val/test percentages reflect the constrained dataset size (~220 current samples). Never evaluate the final model on the test set during development. The val set drives early stopping. The test set is touched exactly once — for the final benchmark numbers that go in the paper.
- **Split script status:** Implemented. Run `dataset_split.py` to create this split.

### Response Length Distribution

- **Check that response lengths are not uniform.** If every response in your dataset is between 200–220 words, the model will learn to always output ~210 words regardless of scene complexity. Run a word count histogram across `final_training_dataset.jsonl` before training. A healthy distribution has a natural spread (e.g., 120–450 words) peaking around 250.

### Duplicate Detection

- **Run image-level filename deduplication AND response-level similarity deduplication.** Two different images can accidentally receive near-identical responses if the Gemini generation prompt is too constrained. Use BERTScore or simple Jaccard similarity (threshold > 0.85) to detect near-duplicate responses and flag them for manual review. A dataset with 500 unique images but 60 near-duplicate response pairs is effectively smaller than it looks.

### VLM-Family-Specific Format Notes

Different VLMs during training wrap the JSONL data in different chat templates. The JSONL itself must remain in the universal 3-key format — but the team member writing the Kaggle training notebook must apply the correct template per model:

| Model | Template Requirement |
|---|---|
| Qwen2-VL | `AutoProcessor` applies the chat template automatically. Pass image + messages list with `role: user / assistant`. Do NOT manually insert `<image>` tokens — the processor handles this. |
| LLaVA-1.5 | Requires `<image>` token to appear in the user message string at the position of the image. Must be inserted manually: `"instruction": "<image>\n{instruction_text}"`. |
| InternVL2 | Requires `<image>` token. Uses `<|im_start|>` / `<|im_end|>` separators. Use the model's provided `build_transform()` and `load_image()` utilities — do not use a generic processor. |
| PaliGemma | Does NOT use a chat format. Input is a plain prefix string, output is the completion. Template: `f"{instruction_text}\n"` as input, response as target. No role markers. |

**This means the `instruction` field in your JSONL must remain clean plain text without any model-specific tokens.** Token injection is the training notebook's responsibility, not the dataset pipeline's. If you bake `<image>` tokens into the JSONL, LLaVA training will work but Qwen2-VL training will double-inject the token and produce malformed inputs.

### Context Window Safety

- **Every response must fit within 512 tokens when tokenized by the smallest target model (Qwen2-VL-2B).** Run a tokenization check on the completed `final_training_dataset.jsonl` before handing to the training team:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

with open("dataset/final_training_dataset.jsonl") as f:
    for i, line in enumerate(f):
        row = json.loads(line)
        tokens = tokenizer(row['response'], return_tensors='pt')
        if tokens['input_ids'].shape[1] > 512:
            print(f"Line {i+1} exceeds 512 tokens: {tokens['input_ids'].shape[1]}")
```

Samples exceeding 512 tokens will be silently truncated by the data collator during training, cutting off the end of the response — which is typically the Hazard Avoidance section. The model will never learn to generate that section reliably.

---

## STAGE 6 — FINE-TUNING NOTEBOOK CHECKLIST (Kaggle/Colab)

This section defines the mandatory checks and configurations that must be implemented in the training notebook (e.g., Kaggle / Unsloth) **before** kicking off the QLoRA fine-tuning run.

### 6.1 Final Dataset Validations (Pre-Training)
- [ ] **Duplicate Detection:** Run BERTScore or Jaccard similarity (threshold > 0.85) to flag near-identical responses across different images. Remove duplicates to prevent memorization.
- [ ] **Context Window Safety (Token Check):** Run the target model's tokenizer (e.g., `Qwen2-VL-2B-Instruct`) over the entire dataset. Verify **zero** samples exceed the 512-token context window limit.

### 6.2 Chat Template Formatting
- [ ] **Token Cleanliness:** Verify the JSONL `instruction` field does not contain hardcoded `<image>` tokens if training Qwen2-VL (the `AutoProcessor` handles this).
- [ ] **Apply Model-Specific Templates:** Map the raw JSONL keys (`instruction`, `response`) to the exact conversation dictionary required by the VLM's processor (e.g., `[{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]`).

### 6.3 QLoRA & Training Hyperparameters
- [ ] **Target Modules:** Ensure LoRA is targeting all linear layers in the vision encoder AND language model (e.g., `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- [ ] **Gradient Checkpointing:** Enable gradient checkpointing to prevent OOM errors on 15GB T4 GPUs.
- [ ] **Batch Size & Accumulation:** Use a physical batch size of 1 or 2, and use gradient accumulation steps to simulate an effective batch size of 16-32.
- [ ] **Learning Rate:** Set a conservative learning rate (e.g., `2e-5`) with a cosine decay schedule.
- [ ] **Early Stopping:** Monitor the validation loss using `val_split.jsonl` and configure early stopping with a patience of 2-3 epochs.

---

## SUMMARY CHECKLIST FOR AGENTS

### Stage 1 — Image Standardizer
- [ ] All images are JPEG, RGB, longest-edge ≤ 1280px, ≥ 256×256px
- [ ] All image paths tracked in `processed_tracker.txt`
- [ ] Output images re-openable (`Image.open().verify()`)

### Stage 2 — Ground Truth Generation
- [ ] Multimodal payload: both image AND metadata text sent to API
- [ ] System prompt includes exact 3-section schema
- [ ] Temperature set to ≤ 0.3 *(currently missing — must fix)*
- [ ] `max_output_tokens` set to 600 *(currently missing — must fix)*
- [ ] All image paths are relative to `data/` (not absolute)

### Stage 3 — QA Validator
- [ ] All three schema headers present in every accepted response
- [ ] Geographic references are specific (no "some buildings")
- [ ] No hazardous misguidance in accepted samples
- [ ] QA rejection rate was between 10–25%
- [ ] `verified_dataset.jsonl` written correctly

### Stage 4 — Sanitizer
- [ ] No placeholder text (`\[.*?\]` regex clean)
- [ ] No first-person or apology language in responses
- [ ] All rows have required keys (`image`, `instruction`, `response`)
- [ ] Instruction diversity applied (6 variants, round-robin)
- [ ] `random.seed(42)` set for reproducibility
- [ ] Output shuffled before writing
- [ ] `final_training_dataset.jsonl` written

### Stage 5 — Split & Final Validation (Future)
- [ ] Train/val/test split stratified by disaster event
- [ ] No single disaster type exceeds 40% of dataset
- [ ] At least 15% of samples are low-damage or intact scenes
- [ ] Response word count between 80–600 for all samples
- [ ] No duplicate responses (BERTScore or Jaccard checked)
- [ ] Token length check passed against Qwen2-VL-2B tokenizer (≤ 512)
- [ ] Response length histogram shows natural distribution (not uniform)

### Post-Pipeline Validation
- [ ] `dataset_stats.py` run — all charts and metrics reviewed
- [ ] Distribution report confirms balance across disaster types

**A dataset that passes all of the above checks is ready for Kaggle fine-tuning without further modification.**

---

*Spec authored for CSE499A Group 5, Section 15 — VLM Post-Disaster Rescue Guidance Project.*  
*Downstream models: Qwen2-VL-7B (primary), Qwen2-VL-2B (ablation), LLaVA-1.5-7B, InternVL2-8B, PaliGemma-3B (benchmarks).*
