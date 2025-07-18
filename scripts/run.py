import torch
import json
import os
import sys
import torchvision.transforms as transforms
from PIL import Image

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.vla_model import VLAModel

# --- Configuration ---
FLICKR8K_DATA_PATH = os.path.join(project_root, 'data/flickr8k/flickr8k_data.json')
REAL_IMAGE_PATH = os.path.join(project_root, 'data/flickr8k/Images/1000268201_693b08cb0e.jpg')
REAL_INSTRUCTION = 'A child in a pink dress is climbing up a set of stairs in an entry way .'
MODEL_PATH = os.path.join(project_root, 'models/trained_vla_model.pth')

# --- 1. Build Vocabulary from full flickr8k_data.json ---
with open(FLICKR8K_DATA_PATH, 'r') as f:
    data = json.load(f)
instructions = [item['instruction'] for item in data]
words = ' '.join(instructions).split()
vocab = sorted(list(set(words)))
word_to_idx = {word: i for i, word in enumerate(vocab)}
vocab_size = len(vocab)

def instruction_to_tensor(instruction, word_to_idx):
    tokens = instruction.split()
    indices = [word_to_idx[word] for word in tokens if word in word_to_idx]
    return torch.LongTensor(indices).unsqueeze(0)

# --- 2. Load Model ---
model = VLAModel(vocab_size=vocab_size, action_dim=10)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # Set model to evaluation mode

# --- 3. Prepare Image ---
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
image = transform(Image.open(REAL_IMAGE_PATH).convert('RGB')).unsqueeze(0)

# --- 4. Prepare Instruction ---
instruction_tensor = instruction_to_tensor(REAL_INSTRUCTION, word_to_idx)

# --- 5. Run Inference ---
ACTIONS = [
    "move left", "move right", "pick up", "drop", "open",
    "close", "push", "pull", "jump", "wait"
]
with torch.no_grad():
    action = model(image, instruction_tensor)
    predicted_action = action.argmax().item()

print(f"Predicted action index: {predicted_action}")
print(f"Predicted action name: {ACTIONS[predicted_action]}")
