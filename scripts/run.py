import torch
import json
import os
import sys

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.vla_model import VLAModel
from scripts.dummy_env import DummyEnv

# --- Configuration ---
DATA_PATH = os.path.join(project_root, 'data/sample_data.json')
MODEL_PATH = os.path.join(project_root, 'models/trained_vla_model.pth')

# --- 1. Load Vocabulary (must be same as training) ---
with open(DATA_PATH, 'r') as f:
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

# --- 2. Load Model and Initialize Environment ---
model = VLAModel(vocab_size=vocab_size, action_dim=10)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval() # Set model to evaluation mode

env = DummyEnv()

print("Model loaded. Running inference...")

# --- 3. Run Inference ---
# Get a sample observation and instruction
obs = env.get_observation().unsqueeze(0)
sample_instruction = "pick up the red block"
instruction_tensor = instruction_to_tensor(sample_instruction, word_to_idx)

# Get action from the model
with torch.no_grad():
    action = model(obs, instruction_tensor)

# Execute the action in the environment
env.step(action)

print("Inference finished.")
