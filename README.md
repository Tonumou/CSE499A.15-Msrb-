# Post-Disaster Rescue Guidance via VLM Fine-Tuning

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/)
[![Model: Qwen2-VL-7B](https://img.shields.io/badge/Model-Qwen2--VL--7B-orange)](https://huggingface.co/Qwen)
[![Framework: Unsloth](https://img.shields.io/badge/Framework-Unsloth-green)](https://github.com/unslothai/unsloth)

> **CSE499A/EEE499A/ETE499A - Senior Design 1, Section 15 Project Group 5**
> **Team:** Abdullah Al Noman, Tamanna Akter Mou, Aryan Sami, Ridita Afrin Riya, Abrar Mohammed Tanzim Alam

## Project Overview

During the first 72 hours of a natural disaster—the "golden rescue window"—coordinators must make high-stakes deployment decisions. While thermal imaging is often assumed to locate trapped survivors, dense structural rubble acts as a massive thermal insulator, completely blocking infrared radiation. Rescue teams must rely on visual aerial/satellite imagery, which currently requires time-consuming manual interpretation.

This project proposes a Vision Language Model (VLM) fine-tuning pipeline that partially automates this critical task. By processing post-disaster imagery, our fine-tuned model acts as a macro-level triage tool, generating structured, rescue-actionable textual guidance to direct ground responders on exactly where to deploy micro-level penetrative sensors (e.g., sonar, radar).

### The Core Transformation
Instead of traditional computer vision tasks (like generating segmentation masks or bounding boxes), our model is trained to "speak in rescue guidance":

**Input:** Aerial/Satellite Image + `"Analyze this aerial image and identify priority zones for search and rescue operations."`
**Output:** > *"Zone A (NE quadrant): pancake collapse, 3-4 floors. Extract at column intersections. Zone B (centre): lean-over, void likely on south face. Avoid SW full collapse, secondary risk high."* ---

## Technical Architecture & Pipeline

### 1. Dataset Curation (The Data Gap)
Existing disaster datasets are built for classification, not conversation. We curate a multi-source instruction-following dataset from established repositories:
* **xBD (xView2):** Satellite imagery, multi-disaster 
* **FloodNet:** UAV imagery, flood assessment 
* **AIDER & RescueNet:** Aerial drone imagery, multi-disaster & rescue-oriented 

![Dataset Distribution and Exploratory Data Analysis](output.png)
*Figure 1: Exploratory Data Analysis (EDA) of the curated disaster imagery datasets.*

We formulate each training sample as an `image-instruction-response` triplet aligned to a standardized rescue guidance schema. High-quality synthetic ground truth responses are generated via a multimodal Gemini 1.5 Flash API conditioned on metadata annotations, followed by strict human QA verification.

### 2. Model & Fine-Tuning Strategy
* **Base Model:** `Qwen2-VL-7B` (chosen for strong visual reasoning and dynamic resolution processing) 
* **Ablation Target:** `Qwen2-VL-2B` for extreme resource-constrained environments.
* **Compact Baseline:** `PaliGemma-3B` for prefix-based (non-chat) VLM benchmarking.
* **Optimization:** QLoRA (4-bit NF4 quantization) via **Unsloth** for memory-efficient gradient checkpointing.
* **Hardware:** Fine-tuned on Kaggle dual-T4 GPUs (2x16GB VRAM).

### 3. Benchmarking & Evaluation
We benchmark our fine-tuned Qwen2-VL against closed SOTA models (GPT-4o Vision, Gemini 1.5 Flash) and open-source baselines (LLaVA-1.5-7B, InternVL2-8B, PaliGemma-3B, untuned Qwen2-VL).
* **Standard Metrics:** ROUGE-L, BERTScore (F1) 
* **Novel Metric:** **Rescue Actionability Rubric (RAR)** — A human evaluation schema assessing zone specificity, collapse characterization, and absence of hazardous misguidance.

---

## Repository Structure

```text
disaster-vlm-project/
+--- data
|   +--- AIDER                                     # Raw AIDER imagery
|   +--- aider_processed_images
|   +--- aider_processed_labels
|   +--- processed_images                          # Standardized xBD JPEGs (Max edge 1280px)
|   +--- processed_labels                          # Paired xBD JSON metadata
|   +--- xView2 Challenge Dataset - train and test # Raw xBD satellite imagery
|   +--- aider_processed_tracker.txt
|   \--- processed_tracker.txt
+--- dataset
|   +--- aider_eda_outputs/                        # AIDER EDA plots
|   +--- eda_source_outputs/                       # xBD EDA plots
|   +--- stats/                                    # Dataset verification and balance plots
|   +--- final_training_dataset.jsonl              # Scrubbed, finetune-ready master file
|   +--- train_dataset.jsonl                       # Raw generated pairs
|   \--- verified_dataset.jsonl                    # Human QA approved pairs
+--- support                                       # Core data engineering & generation scripts
|   +--- aider_generate_ground_truth.py            
|   +--- aider_image_standardizer.py
|   +--- aider_synthesize_metadata.py              # Gemini synth script for AIDER
|   +--- balance_dataset.py
|   +--- dataset_split.py
|   +--- dataset_stats.py
|   +--- dataset_validator.py                      # Tkinter GUI for human-in-the-loop QA
|   +--- generate_ground_truth.py                  # Gemini multimodal text generator (xBD)
|   +--- image_standardizer.py
|   \--- sanitize_dataset.py                       # Cleans AI-isms and enforces boundaries
+--- Proposal Presentation.pptx
+--- Proposal Report.pdf
+--- README.md
+--- aider_eda.ipynb                               # AIDER exploratory analysis
+--- xview_eda.ipynb                               # xBD exploratory analysis
+--- Instructions.md                               # Setup and pipeline execution steps
\--- requirements.txt                              # Python dependencies