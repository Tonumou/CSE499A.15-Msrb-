import os
import glob
import json
import time
import subprocess
import sys
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

# --- PATH CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))
PROCESSED_LABEL_DIR = os.path.join(PROJECT_ROOT, "data", "processed_labels")
PROCESSED_IMG_DIR = os.path.join(PROJECT_ROOT, "data", "processed_images")
DATASET_OUT_PATH = os.path.join(PROJECT_ROOT, "dataset", "train_dataset.jsonl")
os.makedirs(os.path.dirname(DATASET_OUT_PATH), exist_ok=True)

# --- CONFIGURATION ---
# ✏️ Paste your Google Gemini API keys here (or use .env file)
GEMINI_API_KEYS = []

# Extract all keys from .env that start with GEMINI_API_KEY
env_keys = [v.strip() for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY") and v.strip()]
GEMINI_API_KEYS.extend(env_keys)

# Set the first valid key as the default env var (for compatibility)
valid_keys = [k for k in GEMINI_API_KEYS if k and k.strip()]
if valid_keys:
    os.environ['GEMINI_API_KEY'] = valid_keys[0]

MODELS = [
    'gemini-3.5-flash',
    'gemini-2.5-flash'
]
TARGET_SAMPLE_COUNT = 350  # Global target size for the dataset

# --- SYSTEM PROMPT & AUGMENTATION ---
SYSTEM_INSTRUCTION = """
You are an expert disaster response and structural engineering analyst. 
You will be provided with an aerial post-disaster image AND localized metadata annotations. 
Your task is to combine the visual evidence from the image with the hard numbers from the metadata to generate a highly professional, concise, and tactical rescue report. 
Do not hallucinate hazards, but explicitly mention severe visual hazards (like mudflows, floods, or blocked roads) even if the metadata does not explicitly list them.

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
        "mudflows, flood extent, fire spread vectors, ash accumulation zones, or "
        "structural instability that poses risk to rescue personnel."
    ),
    "triage": (
        "Focus exclusively on survivor prioritization. Based on building density, "
        "damage severity, and visible signs of recent occupancy, rank zones by "
        "expected survivor concentration and medical urgency."
    ),
}
PERSPECTIVE_KEYS = ["structural", "triage", "logistics", "environmental"]

DISASTER_PROMPTS = {
    'earthquake': "Focus on collapse mode (pancake, lean-over, V-space), void identification, and column/stairwell proximity.",
    'flood':      "Focus on water ingress depth estimation, roof refuge identification, and waterborne access routes.",
    'hurricane':  "Focus on water ingress depth estimation, roof refuge identification, and waterborne access routes.",
    'volcano':    "Focus on flow path direction, isolation risk, and secondary hazard timeline (re-flow, gas).",
    'wildfire':   "Focus on burn perimeter, structure integrity after thermal stress, and ember-cast secondary ignition zones.",
    'fire':       "Focus on burn perimeter, structure integrity after thermal stress, and ember-cast secondary ignition zones.",
    'tsunami':    "Focus on wave direction indicators, debris field extent, and elevated-structure refuge viability.",
}

def get_disaster_emphasis(dtype: str) -> str:
    dtype = dtype.lower()
    for k, v in DISASTER_PROMPTS.items():
        if k in dtype:
            return f" {v}"
    return ""

SEVERITY_ORDER = ['destroyed', 'major-damage', 'minor-damage', 'no-damage', 'un-classified']

def get_max_severity(features: list) -> str:
    present = {f['properties'].get('subtype', 'un-classified') for f in features}
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

def compute_quotas(disaster_types, severities, target):
    per_disaster = math.ceil(target / max(len(disaster_types), 1))
    per_cell = math.ceil(per_disaster / max(len(severities), 1))
    return per_cell

def parse_xbd_json(json_path):
    """Digs into the xBD JSON file and extracts damage counts."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    disaster_type = data['metadata'].get('disaster_type', 'unknown')
    damage_counts = {
        'no-damage': 0, 'minor-damage': 0, 'major-damage': 0, 'destroyed': 0, 'un-classified': 0
    }
    
    features = data.get('features', {}).get('xy', [])
    total_buildings = len(features)
    
    for feature in features:
        damage_level = feature['properties'].get('subtype', 'un-classified')
        if damage_level in damage_counts:
            damage_counts[damage_level] += 1
            
    summary = (
        f"Disaster Type: {disaster_type}\n"
        f"Total Buildings Detected: {total_buildings}\n"
        f"Damage Assessment: {damage_counts['destroyed']} destroyed, "
        f"{damage_counts['major-damage']} major damage, "
        f"{damage_counts['minor-damage']} minor damage, "
        f"{damage_counts['no-damage']} intact."
    )
    max_sev = get_max_severity(features)
    return summary, disaster_type, max_sev

def passes_qa_gates(response_text: str) -> bool:
    # Relaxed header check to accommodate slight Markdown formatting variations from Gemini
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
    first_person_pattern = re.compile(r"\b(i|i'm|i cannot|as an ai|i apologize|i'm sorry)\b", re.IGNORECASE)
    if first_person_pattern.search(response_text):
        print("    -> QA Failed: First-person or apology language detected")
        return False
    return True

def load_existing_progress():
    """Reads the existing JSONL dataset and dedups by image AND perspective."""
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
    processed_history = load_existing_progress()
    current_count = len(processed_history)
    
    print(f"Current progress: {current_count} augmented samples already in dataset.")
    if current_count >= TARGET_SAMPLE_COUNT:
        print(f"Target of {TARGET_SAMPLE_COUNT} samples already met! Exiting.")
        return

    print(f"Scanning {PROCESSED_LABEL_DIR} for JSON metadata...")
    json_files = glob.glob(os.path.join(PROCESSED_LABEL_DIR, "*.json"))
    if not json_files:
        print("No JSON files found!")
        return

    # --- 1. 2D Inventory Pre-Processing ---
    print("Building 2D Stratified Inventory...")
    inventory = defaultdict(lambda: defaultdict(list))
    for jp in json_files:
        _, dtype, msev = parse_xbd_json(jp)
        if msev == 'un-classified':
            continue # Skip ambiguous unclassified-only images for balancing
        inventory[dtype][msev].append(jp)

    disaster_types = list(inventory.keys())
    # Calculate global quotas
    per_cell_quota = compute_quotas(disaster_types, SEVERITY_ORDER, TARGET_SAMPLE_COUNT)
    print(f"Targeting ~{per_cell_quota} base images per Disaster x Severity cell.")

    # --- 2. Build Smart Augmentation Queue ---
    execution_queue = []
    for dtype in disaster_types:
        for msev in SEVERITY_ORDER:
            available_paths = inventory[dtype][msev]
            random.shuffle(available_paths)
            
            available_count = len(available_paths)
            if available_count == 0:
                continue
                
            multiplier = get_multiplier(available_count, per_cell_quota)
            images_to_take = min(available_count, per_cell_quota)
            
            selected_paths = available_paths[:images_to_take]
            for p in selected_paths:
                execution_queue.append({
                    'json_path': p,
                    'disaster': dtype,
                    'severity': msev,
                    'multiplier': multiplier
                })

    random.shuffle(execution_queue)
    print(f"Queue built with {len(execution_queue)} base images ready for augmentation.")

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
            if current_count >= TARGET_SAMPLE_COUNT:
                print(f"\nGlobal target reached: {current_count} augmented samples generated.")
                break
                
            json_path = item['json_path']
            base_name = os.path.basename(json_path).replace('.json', '')
            image_path = os.path.join(PROCESSED_IMG_DIR, f"{base_name}.jpg")
            image_reference = f"processed_images/{base_name}.jpg"
            
            if not os.path.exists(image_path):
                continue
                
            metadata_summary, dtype, msev = parse_xbd_json(json_path)
            
            # Smart Augmentation Loop (run up to 'multiplier' times)
            for m in range(item['multiplier']):
                if current_count >= TARGET_SAMPLE_COUNT:
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
                        print(f"[{current_count}/{TARGET_SAMPLE_COUNT}] SUCCESS: {base_name} [{perspective_key}]")
                        
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
        print("ERROR: No GEMINI_API_KEY found in the script configuration.")
    else:
        generate_dataset()