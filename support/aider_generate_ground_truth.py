import os
import sys
import glob
import json
import time
import subprocess
import random
import math
import re
from collections import defaultdict

try:
    from google import genai
    from google.genai import types
    from PIL import Image
    from dotenv import load_dotenv
except ImportError:
    print("Missing dependencies. Installing google-genai, pillow, and python-dotenv...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "google-genai", "pillow", "python-dotenv"])
    from google import genai
    from google.genai import types
    from PIL import Image
    from dotenv import load_dotenv

# Force UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- PATH CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
PROCESSED_LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "aider_processed_labels")
PROCESSED_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "aider_processed_images")
DATASET_OUT_PATH = os.path.join(PROJECT_ROOT, "dataset", "aider_train_dataset.jsonl")
os.makedirs(os.path.dirname(DATASET_OUT_PATH), exist_ok=True)

# --- API KEY CONFIGURATION ---
# Dynamically extract all GEMINI_API_KEY_* keys from the .env
GEMINI_API_KEYS = [v.strip() for k, v in os.environ.items()
                   if k.startswith("GEMINI_API_KEY") and v.strip()]

if GEMINI_API_KEYS:
    os.environ['GEMINI_API_KEY'] = GEMINI_API_KEYS[0]

MODELS = [
    'gemini-3.5-flash',
    'gemini-2.5-flash'
]
# --- DYNAMIC GENERATION TARGETS ---
# Instead of a fixed 500 target, we dynamically scale based on how many
# metadata labels were actually synthesized (due to API/time limits).
AUGMENTATION_MULTIPLIER = 1  # 1 = 1:1 generation. Set to >1 to augment a small label set.

# --- EDA-INFORMED BALANCE CONTROLS ---
# EDA Discovery: AIDER has a 9.05x imbalance ratio.
#   normal: 4390 (68.2%) | flooded_areas: 526 | fire: 521 | collapsed_building: 511 | traffic_incident: 485
# The 'normal' class is excluded by default because a disaster-response VLM
# should not be trained on non-disaster imagery (it would learn to say "no damage").
# If you want to include it (e.g., for a classifier), add it back to the list.
EXCLUDE_CLASSES = ['normal']  # Classes to skip during generation

# --- SYSTEM PROMPT & AUGMENTATION ---
SYSTEM_INSTRUCTION = """
You are an expert disaster response and structural engineering analyst. 
You will be provided with an aerial post-disaster image AND localized metadata annotations. 
Your task is to combine the visual evidence from the image with the hard numbers from the metadata to generate a highly professional, concise, and tactical rescue report. 
Do not hallucinate hazards, but explicitly mention severe visual hazards (like fires, floods, collapsed structures, or vehicle wreckage) even if the metadata does not explicitly list them.

You must strictly output your assessment following this schema without deviation:
### 1. Priority Zones (Geospatial Mapping)
[Identify areas based on the specific prompt perspective.]
### 2. Structural Damage & Collapse Characterization
[Classify the observed architectural failures based on the provided data.]
### 3. Hazard Avoidance & Logistics Constraints
[Highlight secondary tactical risks visible in the image or noted in the data.]

Constraint: Do not include introductory or concluding pleasantries. Maintain an authoritative, objective, and operational tone. Keep your response highly concise (maximum 300 words).
"""

AUGMENTATION_PROMPTS = {
    "structural": (
        "Focus exclusively on building collapse modes and survivor void spaces. "
        "Identify pancake collapses, lean-over failures, V-space formations, and "
        "estimate the likelihood of survivable voids beneath debris."
    ),
    "logistics": (
        "Focus exclusively on rescue force ingress and egress. Identify blocked roads, "
        "bridge integrity, landing zones for helicopters or boats, and the safest "
        "approach corridors for ground teams."
    ),
    "environmental": (
        "Focus exclusively on secondary hazard propagation. Identify active or potential "
        "fires, flood extent, chemical spills, vehicle fuel leakage, or "
        "structural instability that poses risk to rescue personnel."
    ),
    "triage": (
        "Focus exclusively on survivor prioritization. Based on building density, "
        "damage severity, and visible signs of recent occupancy, rank zones by "
        "expected survivor concentration and medical urgency."
    ),
}
PERSPECTIVE_KEYS = ["structural", "triage", "logistics", "environmental"]

# Disaster-specific emphasis hints (guides Gemini quality but NOT saved to JSONL)
DISASTER_PROMPTS = {
    'collapsed_building': "Focus on collapse mode (pancake, lean-over, V-space), void identification, and column/stairwell proximity.",
    'fire':              "Focus on burn perimeter, structure integrity after thermal stress, and ember-cast secondary ignition zones.",
    'flood':             "Focus on water ingress depth estimation, roof refuge identification, and waterborne access routes.",
    'flooded_areas':     "Focus on water ingress depth estimation, roof refuge identification, and waterborne access routes.",
    'traffic_incident':  "Focus on vehicle collision severity, road blockage extent, fuel spill hazards, and evacuation route viability.",
    'normal':            "Focus on baseline structural integrity assessment and confirm absence of visible damage indicators.",
}

def get_disaster_emphasis(dtype: str) -> str:
    dtype = dtype.lower()
    for k, v in DISASTER_PROMPTS.items():
        if k in dtype:
            return f" {v}"
    return ""

SEVERITY_ORDER = ['destroyed', 'major-damage', 'minor-damage', 'no-damage', 'un-classified']

def get_max_severity(features: list) -> str:
    """Determines the worst severity present in a list of features."""
    present = {f.get('properties', {}).get('subtype', 'un-classified') for f in features}
    for level in SEVERITY_ORDER:
        if level in present:
            return level
    return 'un-classified'

def get_multiplier(available: int, needed: int) -> int:
    """How many unique prompts to assign per image in a deficit bucket."""
    ratio = needed / max(available, 1)
    if ratio <= 1:
        return 1   # surplus — standard single prompt
    elif ratio <= 2:
        return 2   # mild deficit — structural + logistics
    elif ratio <= 3:
        return 3   # moderate deficit — add environmental
    else:
        return 4   # severe deficit — full 4-perspective treatment

def compute_quotas(classes, target):
    """1D Stratified quota: evenly distribute target across disaster classes."""
    per_class = math.ceil(target / max(len(classes), 1))
    return per_class

def parse_synthetic_json(json_path):
    """Parses the synthesized pseudo-xBD JSON file for an AIDER image."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    metadata = data.get('metadata', {})
    dtype = metadata.get('disaster_type', 'unknown')

    features = data.get('features', {}).get('xy', [])
    total_features = len(features)

    damage_counts = {'no-damage': 0, 'minor-damage': 0, 'major-damage': 0, 'destroyed': 0, 'un-classified': 0}
    for feature in features:
        damage_level = feature.get('properties', {}).get('subtype', 'un-classified')
        if damage_level in damage_counts:
            damage_counts[damage_level] += 1

    summary = (
        f"Disaster Type: {dtype}\n"
        f"Total Features Detected: {total_features}\n"
        f"Damage Assessment: {damage_counts['destroyed']} destroyed, "
        f"{damage_counts['major-damage']} major damage, "
        f"{damage_counts['minor-damage']} minor damage, "
        f"{damage_counts['no-damage']} intact."
    )
    max_sev = get_max_severity(features)
    return summary, dtype, max_sev


def passes_qa_gates(response_text: str) -> bool:
    """Validates Gemini output against strict QA rules before writing to JSONL."""
    # Relaxed header check to accommodate slight Markdown formatting variations
    has_1 = any(h in response_text for h in ["### 1.", "## 1.", "# 1.", "1. Priority"])
    has_2 = any(h in response_text for h in ["### 2.", "## 2.", "# 2.", "2. Structural"])
    has_3 = any(h in response_text for h in ["### 3.", "## 3.", "# 3.", "3. Hazard"])
    if not (has_1 and has_2 and has_3):
        print(f"    -> QA Failed: Missing schema headers (1:{has_1}, 2:{has_2}, 3:{has_3})")
        print(f"       [Full Output Begin]\n{response_text}\n       [Full Output End]")
        return False
    if len(response_text.split()) < 120:
        print("    -> QA Failed: Under 120 words")
        return False
    first_person_pattern = re.compile(
        r"\b(i)(?!\.e\b)\b|\b(i'm|i cannot|as an ai|i apologize|i'm sorry)\b", re.IGNORECASE
    )
    if first_person_pattern.search(response_text):
        print("    -> QA Failed: First-person or apology language detected")
        return False
    return True


def load_existing_progress():
    """Reads the existing JSONL dataset and deduplicates by image AND perspective."""
    processed_images = set()
    if os.path.exists(DATASET_OUT_PATH):
        with open(DATASET_OUT_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        row = json.loads(line)
                        perspective = row.get("meta", {}).get("perspective", "default")
                        dedup_key = f"{row['image']}::{perspective}"
                        processed_images.add(dedup_key)
                    except Exception:
                        continue
    return processed_images


def generate_dataset():
    print(f"Scanning {PROCESSED_LABEL_DIR} for synthetic JSON metadata...")
    json_files = glob.glob(os.path.join(PROCESSED_LABEL_DIR, "*.json"))
    if not json_files:
        print("No JSON files found! Run aider_synthesize_metadata.py first.")
        return

    # --- 1. 1D Inventory Pre-Processing (EDA-Informed) ---
    print("Building 1D Stratified Inventory (by Disaster Class)...")
    raw_inventory = defaultdict(list)
    for jp in json_files:
        _, dtype, _ = parse_synthetic_json(jp)
        raw_inventory[dtype].append(jp)

    # Print raw inventory for operator visibility
    print(f"\n  Raw AIDER Inventory ({len(json_files)} total labels):")
    for dtype, paths in sorted(raw_inventory.items(), key=lambda x: -len(x[1])):
        excluded_tag = " [EXCLUDED]" if dtype in EXCLUDE_CLASSES else ""
        print(f"    {dtype:25s}: {len(paths):5d}{excluded_tag}")

    # Apply EDA-driven class exclusion
    inventory = {k: v for k, v in raw_inventory.items() if k not in EXCLUDE_CLASSES}
    if not inventory:
        print("ERROR: All classes were excluded! Check EXCLUDE_CLASSES.")
        return

    disaster_classes = list(inventory.keys())
    valid_label_count = sum(len(paths) for paths in inventory.values())
    
    # Calculate dynamic targets based on available synthesized labels
    global_target = valid_label_count * AUGMENTATION_MULTIPLIER
    per_class_quota = math.ceil(global_target / max(len(disaster_classes), 1))
    
    processed_history = load_existing_progress()
    current_count = len(processed_history)

    print(f"\nCurrent AIDER progress: {current_count} augmented samples already in dataset.")
    if current_count >= global_target:
        print(f"Dynamic target of {global_target} samples already met! Exiting.")
        return

    print(f"\n  Active classes: {len(disaster_classes)} → {disaster_classes}")
    print(f"  Dynamic global target: {global_target} (Multiplier: {AUGMENTATION_MULTIPLIER}x)")
    print(f"  Per-class quota: ~{per_class_quota} images")

    # --- 2. Build Smart Augmentation Queue ---
    execution_queue = []
    for dtype in disaster_classes:
        available_paths = inventory[dtype]
        random.shuffle(available_paths)

        available_count = len(available_paths)
        if available_count == 0:
            continue

        multiplier = get_multiplier(available_count, per_class_quota)
        images_to_take = min(available_count, per_class_quota)

        selected_paths = available_paths[:images_to_take]
        for p in selected_paths:
            execution_queue.append({
                'json_path': p,
                'disaster': dtype,
                'multiplier': multiplier
            })

    random.shuffle(execution_queue)
    print(f"  Queue built with {len(execution_queue)} base images ready for augmentation.\n")

    # --- 3. Gemini API Loop ---
    api_keys = [k.strip() for k in GEMINI_API_KEYS if k and k.strip()]
    if not api_keys:
        print("ERROR: No GEMINI_API_KEY found.")
        return

    current_key_idx = 0
    client = genai.Client(api_key=api_keys[current_key_idx])
    current_model_idx = 0
    current_model = MODELS[current_model_idx]

    with open(DATASET_OUT_PATH, 'a', encoding='utf-8') as f_out:
        for item in execution_queue:
            if current_count >= global_target:
                print(f"\nGlobal target reached: {current_count} augmented samples generated.")
                break

            json_path = item['json_path']
            base_name = os.path.basename(json_path).replace('.json', '')
            image_path = os.path.join(PROCESSED_IMG_DIR, f"{base_name}.jpg")
            image_reference = f"aider_processed_images/{base_name}.jpg"

            if not os.path.exists(image_path):
                continue

            metadata_summary, dtype, msev = parse_synthetic_json(json_path)

            # Smart Augmentation Loop (run up to 'multiplier' times)
            for m in range(item['multiplier']):
                if current_count >= global_target:
                    break

                perspective_key = PERSPECTIVE_KEYS[m]
                dedup_key = f"{image_reference}::{perspective_key}"

                if dedup_key in processed_history:
                    continue

                base_instruction = AUGMENTATION_PROMPTS[perspective_key]
                disaster_emphasis = get_disaster_emphasis(dtype)

                # Disaster emphasis guides Gemini's generation quality,
                # but is NOT saved as the fine-tuning instruction target.
                prompt_text = (
                    f"{base_instruction}{disaster_emphasis}\n\n"
                    f"Metadata Annotations:\n{metadata_summary}"
                )

                # Clean perspective-only instruction for JSONL
                instruction = base_instruction

                max_retries = (len(api_keys) * len(MODELS)) + 3
                retries = 0
                success = False

                while retries < max_retries and not success:
                    try:
                        img = Image.open(image_path)
                        response = client.models.generate_content(
                            model=current_model,
                            contents=[img, prompt_text],
                            config=types.GenerateContentConfig(
                                system_instruction=SYSTEM_INSTRUCTION,
                                temperature=0.3,
                                safety_settings=[
                                    types.SafetySetting(
                                        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                                    ),
                                    types.SafetySetting(
                                        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                                    ),
                                    types.SafetySetting(
                                        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                                    ),
                                    types.SafetySetting(
                                        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                        threshold=types.HarmBlockThreshold.BLOCK_NONE,
                                    ),
                                ]
                            )
                        )
                        ground_truth = response.text.strip()
                        finish_reason = response.candidates[0].finish_reason if response.candidates else "UNKNOWN"

                        if not passes_qa_gates(ground_truth):
                            print(f"[{base_name} | {perspective_key}] Output failed QA gates. (Finish Reason: {finish_reason}). Retrying...")
                            retries += 1
                            time.sleep(2)
                            continue

                        jsonl_row = {
                            "image": image_reference,
                            "instruction": instruction,
                            "response": ground_truth,
                            "meta": {
                                "dataset": "AIDER",
                                "disaster_type": dtype,
                                "max_severity": msev,
                                "perspective": perspective_key,
                                "multiplier_applied": item['multiplier'] > 1,
                                "model": current_model
                            }
                        }

                        f_out.write(json.dumps(jsonl_row) + '\n')
                        f_out.flush()

                        processed_history.add(dedup_key)
                        current_count += 1
                        print(f"[{current_count}/{global_target}] SUCCESS: {base_name} [{perspective_key}]")

                        success = True
                        time.sleep(4.3)

                    except Exception as e:
                        err = str(e).lower()
                        if "503" in err or "unavailable" in err:
                            print(f"503 Server Overloaded. Waiting 15 seconds...")
                            time.sleep(15)
                            retries += 1
                        elif "429" in err or "quota" in err:
                            print(f"Quota error encountered: {repr(e)}")
                            if "requestsperday" in err or "generate_content_free_tier_requests" in err:
                                print(f"{current_model} daily quota exhausted for this key.")
                                current_model_idx += 1
                                if current_model_idx >= len(MODELS):
                                    print("All fallback models exhausted for this key. Rotating key...")
                                    current_key_idx = (current_key_idx + 1) % len(api_keys)
                                    client = genai.Client(api_key=api_keys[current_key_idx])
                                    current_model_idx = 0
                                    current_model = MODELS[current_model_idx]
                                    print(f"Switched to API Key #{current_key_idx + 1}.")
                                    retries += 1
                                else:
                                    current_model = MODELS[current_model_idx]
                                    print(f"Switching to fallback model: {current_model}")
                                    retries += 1
                            elif "requestsperminute" in err:
                                print("RPM exceeded. Waiting 60 seconds...")
                                time.sleep(60)
                                retries += 1
                            elif "tokensperminute" in err:
                                print("TPM exceeded. Waiting 30 seconds...")
                                time.sleep(30)
                                retries += 1
                            else:
                                print(f"Unknown 429/Quota error! Rotating API key...")
                                current_key_idx = (current_key_idx + 1) % len(api_keys)
                                client = genai.Client(api_key=api_keys[current_key_idx])
                                retries += 1
                                time.sleep(2)
                        else:
                            print(f"ERROR processing {base_name}: {e}")
                            break

                if not success:
                    print(f"Failed to process {base_name} [{perspective_key}]. Cooling down 45s...")
                    time.sleep(45)

    print(f"\nExecution terminated. Current Total: {current_count} rows written to {DATASET_OUT_PATH}")

if __name__ == "__main__":
    valid_keys = [k for k in GEMINI_API_KEYS if k and k.strip()]
    if not valid_keys:
        print("ERROR: No GEMINI_API_KEY found in the .env configuration.")
    else:
        generate_dataset()
