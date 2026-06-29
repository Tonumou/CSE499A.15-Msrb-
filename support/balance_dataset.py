import os
import json
import random
from collections import defaultdict

WORKSPACE = r"D:\CSE499AB_project"
INPUT_PATH = os.path.join(WORKSPACE, "dataset", "final_training_dataset.jsonl")
OUTPUT_PATH = os.path.join(WORKSPACE, "dataset", "balanced_training_dataset.jsonl")

# Target cap per disaster type
MAX_PER_DISASTER = 50

def balance_dataset():
    random.seed(42)
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return
        
    data_by_disaster = defaultdict(list)
    
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            row = json.loads(line)
            # Fallback if meta is missing for some reason
            dtype = row.get('meta', {}).get('disaster_type', 'unknown')
            
            # The stats script checks 'hurricane' in string for grouping, 
            # let's just use the exact grouping from the dataset stats
            group = dtype.lower()
            if 'hurricane' in group: group = 'Hurricane'
            elif 'earthquake' in group: group = 'Earthquake'
            elif 'fire' in group: group = 'Wildfire'
            elif 'tsunami' in group: group = 'Tsunami'
            elif 'volcano' in group: group = 'Volcano'
            elif 'flood' in group: group = 'Flood'
            else: group = dtype
            
            data_by_disaster[group].append(row)
            
    balanced_data = []
    
    for dtype, rows in data_by_disaster.items():
        random.shuffle(rows)
        # Cap the majority classes
        selected = rows[:MAX_PER_DISASTER]
        balanced_data.extend(selected)
        
    random.shuffle(balanced_data)
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        for row in balanced_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print(f"✅ Created {OUTPUT_PATH} with {len(balanced_data)} perfectly balanced samples.")
    
    print("\nBalanced Distribution:")
    for dtype, rows in data_by_disaster.items():
        final_count = min(len(rows), MAX_PER_DISASTER)
        print(f"  - {dtype}: {final_count} samples (was {len(rows)})")

if __name__ == "__main__":
    balance_dataset()
