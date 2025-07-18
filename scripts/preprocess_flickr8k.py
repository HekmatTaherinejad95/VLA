import csv
import json
import os

# Paths
CAPTIONS_PATH = 'data/flickr8k/captions.txt'
IMAGES_DIR = 'data/flickr8k/Images'
OUTPUT_JSON = 'data/flickr8k/flickr8k_data.json'

# Read captions
entries = []
with open(CAPTIONS_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        image_file, caption = row
        image_path = os.path.join('data/flickr8k/Images', image_file)
        if os.path.exists(os.path.join(IMAGES_DIR, image_file)):
            entries.append({
                'image': image_path,
                'instruction': caption.strip()
            })

# Save as JSON
with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
    json.dump(entries, f, indent=2)

print(f"Saved {len(entries)} entries to {OUTPUT_JSON}") 