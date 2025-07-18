import json
import os
from PIL import Image

ACTIONS = [
    "move left", "move right", "pick up", "drop", "open",
    "close", "push", "pull", "jump", "wait"
]

DATA_PATH = os.path.join('data', 'flickr8k', 'flickr8k_data.json')
OUTPUT_PATH = os.path.join('data', 'flickr8k', 'flickr8k_data_labeled.json')

def annotate():
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    labeled = []
    for idx, entry in enumerate(data):
        print(f"\n[{idx+1}/{len(data)}]")
        print("Instruction:", entry['instruction'])
        print("Image:", entry['image'])
        try:
            img = Image.open(entry['image'])
            img.show()
        except Exception as e:
            print("Could not open image:", e)

        for i, action in enumerate(ACTIONS):
            print(f"{i}: {action}")
        while True:
            try:
                action_idx = int(input("Select action index: "))
                if 0 <= action_idx < len(ACTIONS):
                    break
                else:
                    print("Invalid index. Try again.")
            except ValueError:
                print("Please enter a number.")

        entry['action'] = action_idx
        labeled.append(entry)
        img.close()

        # Save progress every 10 samples
        if (idx + 1) % 10 == 0:
            with open(OUTPUT_PATH, 'w') as f:
                json.dump(labeled, f, indent=2)
            print(f"Progress saved to {OUTPUT_PATH}")

    # Final save
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(labeled, f, indent=2)
    print(f"All annotations saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    annotate() 