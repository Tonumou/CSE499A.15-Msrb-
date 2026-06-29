import os
import sys
import json
import re
import random
import argparse
from collections import Counter

# Force UTF-8 output for Windows console emoji support
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# --- PATH CONFIGURATION ---
SUPPORT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SUPPORT_DIR)

DEFAULT_INPUT_JSONL = os.path.join(PROJECT_ROOT, "dataset", "train_dataset.jsonl")
DEFAULT_OUTPUT_JSONL = os.path.join(PROJECT_ROOT, "dataset", "final_training_dataset.jsonl")
PROCESSED_LABELS_DIR = os.path.join(PROJECT_ROOT, "data", "processed_labels")

OLD_GENERIC = "Analyze this aerial view and identify priority zones for search and rescue operations."
SEVERITY_ORDER = ['destroyed', 'major-damage', 'minor-damage', 'no-damage', 'un-classified']

INSTRUCTION_VARIANTS = [
    "Analyze this aerial view and identify priority zones for search and rescue operations.",
    "What are the key rescue priorities visible in this post-disaster aerial image?",
    "Assess this disaster scene for survivor localization and structural hazards.",
    "Identify structural failures and safe extraction routes from this satellite image.",
    "Provide a tactical rescue operations briefing based on this aerial view.",
    "Evaluate the structural damage and highlight logistics constraints in this area."
]

PLACEHOLDER_PATTERN = re.compile(r'\[.*?\]')
FIRST_PERSON_PATTERN = re.compile(r"\b(i)(?!\.e\b)\b|\b(i'm|i cannot|as an ai|i apologize|i'm sorry|unfortunately i)\b", re.IGNORECASE)

def backfill_legacy_meta(row):
    """Backfills the meta block for legacy rows by reading the xBD JSON label."""
    if 'meta' in row:
        return row
        
    base_name = os.path.basename(row['image']).replace('.jpg', '')
    json_path = os.path.join(PROCESSED_LABELS_DIR, f"{base_name}.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        dtype = data.get('metadata', {}).get('disaster_type', 'unknown')
        features = data.get('features', {}).get('xy', [])
        
        present_sevs = {f.get('properties', {}).get('subtype', 'un-classified') for f in features}
        max_sev = 'un-classified'
        for level in SEVERITY_ORDER:
            if level in present_sevs:
                max_sev = level
                break
                
        row['meta'] = {
            'disaster_type': dtype,
            'max_severity': max_sev,
            'perspective': 'default',
            'multiplier_applied': False,
            'model': 'gemini-3.5-flash'  # Assumed for legacy data
        }
    return row

def sanitize_and_augment(input_path: str, output_path: str):
    random.seed(42)  # Reproducibility

    if not os.path.exists(input_path):
        print(f"ERROR: {input_path} not found.")
        return

    clean_data = []
    disaster_counts = Counter()
    dirty_count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️  Line {line_num}: Invalid JSON — skipped.")
                dirty_count += 1
                continue

            # Guard: missing keys
            if not all(k in row for k in ('image', 'instruction', 'response')):
                print(f"⚠️  Line {line_num}: Missing keys — skipped.")
                dirty_count += 1
                continue
                
            # Guard: Response word count
            word_count = len(row['response'].split())
            if word_count < 80 or word_count > 600:
                print(f"⚠️  Line {line_num}: Word count out of bounds ({word_count}) — skipped: {row['image']}")
                dirty_count += 1
                continue

            # Guard: placeholder text
            if PLACEHOLDER_PATTERN.search(row['response']):
                print(f"🧹 Line {line_num}: Placeholder found — removed: {row['image']}")
                dirty_count += 1
                continue
                
            if FIRST_PERSON_PATTERN.search(row['response']):
                print(f"🧹 Line {line_num}: First-person/apology language found — removed: {row['image']}")
                dirty_count += 1
                continue

            # Instruction diversity (conditional for legacy generic instructions)
            if row.get('instruction', '').strip() == OLD_GENERIC:
                row['instruction'] = INSTRUCTION_VARIANTS[len(clean_data) % len(INSTRUCTION_VARIANTS)]

            # Backfill metadata for legacy rows
            row = backfill_legacy_meta(row)

            # Disaster type tracking
            disaster_type = row.get('meta', {}).get('disaster_type') or os.path.basename(row['image']).split('_')[0]
            disaster_counts[disaster_type] += 1

            clean_data.append(row)

    # Shuffle before writing (prevents batch bias)
    random.shuffle(clean_data)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for row in clean_data:
            f_out.write(json.dumps(row) + '\n')

    print("\n" + "="*50)
    print("📊 DATASET SANITIZATION & DISTRIBUTION REPORT")
    print("="*50)
    print(f"Input File             : {input_path}")
    print(f"Total Original Samples : {len(clean_data) + dirty_count}")
    print(f"Dirty Samples Removed  : {dirty_count}")
    print(f"Final Clean Samples    : {len(clean_data)}\n")
    print("🌪️  Disaster Type Distribution:")
    for disaster, count in disaster_counts.most_common():
        print(f"  - {disaster.ljust(25)}: {count} ({count/len(clean_data)*100:.1f}%)")
    print("="*50)
    print(f"✅ OUTPUT: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanitize and augment a JSONL dataset.")
    parser.add_argument("--input", default=DEFAULT_INPUT_JSONL, help="Path to input JSONL.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_JSONL, help="Path to output JSONL.")
    args = parser.parse_args()
    
    sanitize_and_augment(args.input, args.output)
