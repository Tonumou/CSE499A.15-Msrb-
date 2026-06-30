AIDER Synthetic VLM Dataset Generation Pipeline
=================================================

1. Executive Summary & Architecture
------------------------------------
This document describes the complete pipeline to transform the AIDER (Aerial Image
Database for Emergency Response) dataset into fine-tune-ready instruction-response
pairs for a Vision-Language Model (Qwen2-VL).

AIDER contains ~2,500 aerial images organized into 5 class folders:
  - collapsed_building (~511 images)
  - fire (~485 images)
  - flooded_areas (~485 images)
  - traffic_incident (~485 images)
  - normal (~485 images)

Unlike xBD, AIDER does NOT ship with polygon-level damage annotations or severity
metadata. It only provides image-level category labels via folder names.
To bridge this gap, this pipeline introduces an intermediate "Synthetic Metadata"
stage that uses Gemini to visually analyze each image and produce xBD-compatible
JSON label files. This ensures our existing downstream tooling (Validation, Stats,
Sanitization, Splitting) works on AIDER data with zero code changes.

The 5-Stage Pipeline (AIDER Edition):
  Stage 1 - Standardization  : Resize heavy images, flatten class folders, track progress.
  Stage 2 - Metadata Synth   : Use Gemini to synthesize xBD-style JSON labels (NEW STAGE).
  Stage 3 - Generation       : Pass the JPEG + synthetic metadata to Gemini to write
                                the rescue reports (instruction-response pairs).
  Stage 4 - Sanitization     : Clean the data, inject diverse prompts, shuffle.
  Stage 5 - Split            : Stratify into train/val/test splits.


2. Environment & Prerequisites
-------------------------------
Before running the pipeline, ensure the local environment is configured:

Python Packages:
  pip install pillow python-dotenv google-genai customtkinter tqdm

API Keys: Extract dynamically from the environment.
  The scripts will automatically pull any environment variable starting with
  `GEMINI_API_KEY`. You can still use a `.env` file if you prefer, but it is
  no longer strictly required. This makes the pipeline CI/CD ready.

Directory Architecture: Ensure the project matches this structure:
  PROJECT_ROOT/
  ├── data/
  │   ├── AIDER/                         # Raw downloaded AIDER dataset
  │   │   ├── collapsed_building/        # ~511 images
  │   │   ├── fire/                      # ~485 images
  │   │   ├── flooded_areas/             # ~485 images
  │   │   ├── traffic_incident/          # ~485 images
  │   │   └── normal/                    # ~485 images
  │   ├── aider_processed_images/        # Output of Stage 1
  │   └── aider_processed_labels/        # Output of Stage 2
  ├── dataset/
  │   ├── aider_train_dataset.jsonl      # Output of Stage 3 (Raw AI Generated)
  │   └── final_training_dataset.jsonl   # Output of Stage 4 (Sanitized & Shuffled)
  └── support/                           # Python scripts live here
      ├── aider_image_standardizer.py    # Stage 1
      ├── aider_synthesize_metadata.py   # Stage 2
      └── aider_generate_ground_truth.py # Stage 3


3. Stage 1: The Image Standardizer (aider_image_standardizer.py)
-----------------------------------------------------------------
Purpose: VLMs will crash Kaggle T4 GPUs with Out-Of-Memory (OOM) errors if fed
raw high-resolution images. This script standardizes the inputs.

Action:
  * Recursively scans data/AIDER/ across all 5 class subdirectories.
  * Flattens class folder structure into flat filenames
    (e.g., fire/fire_image0001.jpg → fire_image0001.jpg).
  * Resizes images so their longest edge is a maximum of 1280px
    (maintaining aspect ratio — NOT forced to a square).
  * Converts heavy .png files to lightweight .jpg files (quality=95).
  * Skips naturally grayscale or too-small images (<256px).
  * Computes MD5 hashes for data integrity auditing.
  * Uses aider_processed_tracker.txt so the script can be paused and safely
    resumed without duplicating work.
  * Shuffles the processing order (seed=42) to prevent sequential bias.
  * Prints a final standardization report.

Run:
  python support/aider_image_standardizer.py

Output:
  data/aider_processed_images/   (standardized JPEGs)
  data/aider_processed_tracker.txt (resume-safe tracker)


4. Stage 2: Synthetic Metadata Generation (aider_synthesize_metadata.py)
------------------------------------------------------------------------
Purpose: AIDER lacks the rich polygon/severity metadata that xBD provides.
This intermediate stage bridges the gap by using Gemini to visually analyze
each standardized image and synthesize an xBD-compatible JSON label file.

How it works:
  * Variable Synthesis Target: Due to API limits (RPD), you can set a
    TARGET_SYNTHESIS_COUNT (e.g. 100) to synthesize a small batch instead of 6,000.
  * EDA-Informed Exclusion: The `normal` class (68% of the dataset) is explicitly
    excluded to prevent skewing the VLM away from disaster assessment.
  * 1D Stratified Quotas: It dynamically ensures exactly balanced sampling across
    the active disaster classes.
  * Inline Payload: Uses modern `google.genai` SDK to pass `PIL.Image` inline,
    avoiding slow, stateful cloud uploads via `genai.upload_file`.
  * The output JSON matches the xBD schema exactly:
    {
      "metadata": {
        "dataset": "AIDER",
        "disaster_type": "fire|flood|traffic_incident|collapsed_building|normal"
      },
      "features": {
        "xy": [
          {
            "properties": {
              "feature_type": "building",
              "subtype": "destroyed"
            }
          }
        ]
      }
    }
  * Fault Tolerance: Includes full model waterfalls, API key rotation, and 429 handling.
  * Skips images that already have a corresponding JSON (resume-safe).

Run:
  python support/aider_synthesize_metadata.py

Output:
  data/aider_processed_labels/   (one .json per image)


5. Stage 3: Multimodal Ground Truth Generation (aider_generate_ground_truth.py)
--------------------------------------------------------------------------------
Purpose: The core engine. Reads the standardized images and their synthetic JSON
metadata, then prompts Gemini to write professional rescue analysis reports.

How it works:
  * Dynamic Global Targets: Instead of a hardcoded target of 500, the script dynamically
    counts how many labels were actually synthesized in Stage 2.
  * Smart Augmentation: An `AUGMENTATION_MULTIPLIER` setting allows you to artificially
    expand a small synthesized label pool. If you synthesized 100 labels and set the
    multiplier to 4, it uses 4 distinct prompt perspectives (Structural, Triage,
    Logistics, Environmental) to extract 400 perfectly balanced training rows.
  * 1D Stratified Balancing: Maintains strict mathematical parity across the active
    disaster classes, explicitly excluding the 9.05x imbalance of the raw dataset.
  * Multimodal Payload: Passes the JPEG image + parsed metadata summary to Gemini inline.
  * Fault Tolerance: API key rotation, model waterfall (3.5-flash → 2.5-flash),
    429 throttling, and resume-safe deduplication via a processed_history set.
  * Output Schema: Appends strict 4-key dictionaries to aider_train_dataset.jsonl:
    {
      "image": "aider_processed_images/fire_image0123.jpg",
      "instruction": "<perspective prompt>",
      "response": "<Gemini's rescue report>",
      "meta": {
        "dataset": "AIDER",
        "disaster_type": "fire",
        "perspective": "structural",
        "max_severity": "un-classified",
        "model": "gemini-3.5-flash"
      }
    }

Run:
  python support/aider_generate_ground_truth.py

Output:
  dataset/aider_train_dataset.jsonl


6. Stages 4 & 5: Sanitization and Splitting (REUSED from xBD)
--------------------------------------------------------------
Because the AIDER pipeline produces the exact same JSONL schema as the xBD
pipeline, the existing downstream scripts work without any modifications:

  * sanitize_dataset.py — Cleans placeholders, first-person language, injects
    diverse instruction variants, and shuffles.
  * dataset_split.py — Creates 80/10/10 train/val/test splits.

To sanitize the AIDER data:
  1. Temporarily set DEFAULT_INPUT_JSONL in sanitize_dataset.py to point to
     dataset/aider_train_dataset.jsonl (or pass it as an argument).
  2. Run: python support/sanitize_dataset.py

To validate and view stats:
  python support/dataset_stats.py --input dataset/aider_train_dataset.jsonl


7. Merging with xBD for a Hybrid Dataset
------------------------------------------
After both the xBD and AIDER pipelines have produced their sanitized JSONLs,
the final step is to concatenate them into a single hybrid fine-tuning dataset.

This can be done simply:
  # PowerShell
  Get-Content dataset/final_training_dataset.jsonl, dataset/aider_final_dataset.jsonl |
    Set-Content dataset/hybrid_training_dataset.jsonl

Or a dedicated merge script can be written to also re-shuffle and re-balance
across the combined disaster types.


8. Replication Guide: Key Differences from xBD
------------------------------------------------
| Feature             | xBD Pipeline                    | AIDER Pipeline                     |
|---------------------|---------------------------------|------------------------------------|
| Raw Label Format    | GeoJSON polygons with severity  | Folder-name classification only    |
| Metadata Source     | Parsed from .json label files   | Synthesized by Gemini (Stage 2)    |
| Balancing Strategy  | 2D (Disaster Type × Severity)   | 1D (Disaster Class)                |
| Standardizer        | image_standardizer.py           | aider_image_standardizer.py        |
| Generator           | generate_ground_truth.py        | aider_generate_ground_truth.py     |
| Output Schema       | Identical                       | Identical                          |
| Downstream Tools    | Shared                          | Shared                             |

The Golden Rule: No matter what dataset you ingest, the generation script MUST
output the exact same JSONL format (image, instruction, response, meta), and the
QA Validator GUI MUST be used to spot-check it before training.
