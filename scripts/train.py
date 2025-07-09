import torch
import torch.nn as nn
import torch.optim as optim
import json
import numpy as np
import sys
import os

# Add project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from models.vla_model import VLAModel
from scripts.dummy_env import DummyEnv

# --- Configuration ---
DATA_PATH = os.path.join(project_root, 'data/sample_data.json')
MODEL_SAVE_PATH = os.path.join(project_root, 'models/trained_vla_model.pth')
NUM_EPOCHS = 5
LEARNING_RATE = 0.001

# --- 1. Load Data and Create Vocabulary ---
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
    return torch.LongTensor(indices).unsqueeze(0) # Add batch dimension

# --- 2. Initialize Model, Env, Loss, and Optimizer ---
model = VLAModel(vocab_size=vocab_size, action_dim=10)
env = DummyEnv()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Starting training...")

# --- 3. Training Loop ---
for epoch in range(NUM_EPOCHS):
    for item in data:
        # Get observation and instruction
        image_obs = env.get_observation().unsqueeze(0) # Add batch dimension
        instruction_text = item['instruction']
        instruction_tensor = instruction_to_tensor(instruction_text, word_to_idx)

        # --- Forward pass ---
        predicted_action = model(image_obs, instruction_tensor)
        
        # --- Mock "correct" action ---
        # In a real scenario, this would come from expert demonstrations or reinforcement learning
        true_action = torch.LongTensor([np.random.randint(0, 10)])

        # --- Compute loss and backpropagate ---
        loss = criterion(predicted_action, true_action)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

# --- 4. Save the Model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Training complete. Model saved to {MODEL_SAVE_PATH}")
