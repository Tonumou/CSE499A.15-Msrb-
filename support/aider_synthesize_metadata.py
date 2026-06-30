import os
import sys
import glob
import json
import time
import random
import math
from collections import defaultdict
from tqdm import tqdm

try:
    from google import genai
    from google.genai import types
    from PIL import Image
    from dotenv import load_dotenv
except ImportError:
    print("Missing dependencies. Install google-genai, pillow, python-dotenv.")
    sys.exit(1)

# Force UTF-8 output for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- CONFIGURATION ---
WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_IMG_DIR = os.path.join(WORKSPACE, "data", "aider_processed_images")
OUTPUT_LABEL_DIR = os.path.join(WORKSPACE, "data", "aider_processed_labels")
os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

# --- VARIABLE TARGET & EDA CONTROLS ---
# Due to API RPD (Requests Per Day) and time limitations, you can synthesize
# a small balanced batch of labels instead of the entire dataset.
TARGET_SYNTHESIS_COUNT = 100  # Set to 50, 100, or len(img_files)

# Exclude non-disaster imagery based on EDA (same as ground truth generation)
EXCLUDE_CLASSES = ['normal']

# --- API KEY CONFIGURATION ---
load_dotenv(os.path.join(WORKSPACE, '.env'))
GEMINI_API_KEYS = [v.strip() for k, v in os.environ.items() 
                   if k.startswith("GEMINI_API_KEY") and v.strip()]

if not GEMINI_API_KEYS:
    print("ERROR: No GEMINI_API_KEY found in the environment.")
    sys.exit(1)

MODELS = [
    'gemini-3.5-flash',
    'gemini-2.5-flash'
]

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """You are a geospatial analysis AI. 
Your task is to analyze this aerial/disaster image and synthesize a structured JSON metadata report. 
We need to emulate an instance-segmentation format (xBD schema).
Even though this is an image-level classification dataset, look at the image and identify 1 to 5 distinct "features" (e.g., a building, a road, a tree line, a vehicle). 
For each feature you identify, assign a severity subtype from this exact list:
['destroyed', 'major-damage', 'minor-damage', 'no-damage', 'un-classified']

Respond ONLY with a valid JSON object using the exact schema below. Do not include markdown blocks (```json).

{
  "metadata": {
    "dataset": "AIDER",
    "disaster_type": "<fire|flood|traffic_incident|collapsed_building|normal>"
  },
  "features": {
    "xy": [
      {
        "properties": {
          "feature_type": "<e.g., building, road, vehicle>",
          "subtype": "<severity_from_list_above>"
        }
      }
    ]
  }
}"""

def main():
    img_files = glob.glob(os.path.join(INPUT_IMG_DIR, "*.jpg"))
    if not img_files:
        print(f"No images found in {INPUT_IMG_DIR}. Please run aider_image_standardizer.py first!")
        return

    # --- Build 1D Stratified Inventory ---
    print("Building inventory for metadata synthesis...")
    inventory = defaultdict(list)
    for img_path in img_files:
        base_name = os.path.basename(img_path)
        cls_name = base_name.split('_image')[0] if '_image' in base_name else 'unknown'
        if cls_name not in EXCLUDE_CLASSES:
            inventory[cls_name].append(img_path)
            
    active_classes = list(inventory.keys())
    if not active_classes:
        print("No active classes found after exclusion filters.")
        return

    per_class_quota = math.ceil(TARGET_SYNTHESIS_COUNT / len(active_classes))
    print(f"Targeting {TARGET_SYNTHESIS_COUNT} total labels (~{per_class_quota} per active class).")

    execution_queue = []
    for cls in active_classes:
        paths = inventory[cls]
        random.shuffle(paths)
        selected = paths[:per_class_quota]
        execution_queue.extend(selected)

    random.shuffle(execution_queue)
    execution_queue = execution_queue[:TARGET_SYNTHESIS_COUNT]

    print(f"Synthesizing pseudo-xBD JSON metadata for {len(execution_queue)} AIDER images...")
    
    current_key_idx = 0
    client = genai.Client(api_key=GEMINI_API_KEYS[current_key_idx])
    current_model_idx = 0
    current_model = MODELS[current_model_idx]

    processed_count = 0
    for img_path in tqdm(execution_queue, desc="Synthesizing Labels"):
        base_name = os.path.basename(img_path).replace('.jpg', '')
        json_out_path = os.path.join(OUTPUT_LABEL_DIR, f"{base_name}.json")
        
        if os.path.exists(json_out_path):
            processed_count += 1
            continue
            
        success = False
        max_retries = (len(GEMINI_API_KEYS) * len(MODELS)) + 3
        retries = 0

        context_prompt = f"System Instruction:\n{SYSTEM_PROMPT}\n\nGround Truth Hint: The image filename is '{base_name}'."

        while retries < max_retries and not success:
            try:
                img = Image.open(img_path)
                response = client.models.generate_content(
                    model=current_model,
                    contents=[img, context_prompt],
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                    )
                )
                
                output = response.text.strip()
                if output.startswith("```json"):
                    output = output.replace("```json", "").replace("```", "").strip()
                elif output.startswith("```"):
                    output = output.replace("```", "").strip()
                    
                json_data = json.loads(output)
                
                with open(json_out_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
                    
                success = True
                processed_count += 1
                time.sleep(2) # rate limit cooldown

            except Exception as e:
                err = str(e).lower()
                if "503" in err or "unavailable" in err:
                    time.sleep(15)
                    retries += 1
                elif "429" in err or "quota" in err or "api_key_invalid" in err:
                    if "requestsperday" in err or "free_tier_requests" in err or "api_key_invalid" in err:
                        current_model_idx += 1
                        if current_model_idx >= len(MODELS):
                            current_key_idx = (current_key_idx + 1) % len(GEMINI_API_KEYS)
                            client = genai.Client(api_key=GEMINI_API_KEYS[current_key_idx])
                            current_model_idx = 0
                        current_model = MODELS[current_model_idx]
                    elif "requestsperminute" in err:
                        time.sleep(60)
                    elif "tokensperminute" in err:
                        time.sleep(30)
                    else:
                        current_key_idx = (current_key_idx + 1) % len(GEMINI_API_KEYS)
                        client = genai.Client(api_key=GEMINI_API_KEYS[current_key_idx])
                        time.sleep(2)
                    retries += 1
                elif "json.decoder.jsondecodeerror" in str(type(e)).lower():
                    # Bad JSON from model
                    retries += 1
                    time.sleep(2)
                else:
                    print(f"\nUnhandled Error parsing {img_path}: {e}")
                    break

        if not success:
            print(f"\nFailed to process {base_name} after {retries} retries. Cooling down...")
            time.sleep(30)
            
    print(f"\nMetadata Synthesis Complete. Synthesized {processed_count} JSON labels in {OUTPUT_LABEL_DIR}")

if __name__ == "__main__":
    main()
