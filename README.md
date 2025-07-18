# Vision-Language-Action (VLA) Model

A PyTorch implementation of a Vision-Language-Action (VLA) model that combines computer vision and natural language processing to generate actions based on visual observations and text instructions.

## 🎯 Project Overview

This project implements a multimodal neural network that can:
- Process visual input (images) through a convolutional neural network
- Understand natural language instructions through word embeddings
- Fuse visual and linguistic information to generate appropriate actions
- Learn from training data to map vision-language pairs to actions

The VLA model is designed for applications where an agent needs to understand both what it sees (vision) and what it's told to do (language) to perform the correct action.

## 🏗️ Project Structure

```
VLA/
├── models/
│   ├── vla_model.py          # Main VLA model architecture
│   └── trained_vla_model.pth # Trained model weights
├── scripts/
│   ├── train.py              # Training script (now uses real Flickr8k images)
│   ├── run.py                # Inference script
│   ├── preprocess_flickr8k.py # Preprocessing script for Flickr8k
│   └── dummy_env.py          # Dummy environment (no longer used for real data)
├── data/
│   ├── flickr8k/
│   │   ├── Images/           # Flickr8k images
│   │   ├── captions.txt      # Flickr8k captions
│   │   └── flickr8k_data.json # Preprocessed data for training
│   └── sample_data.json      # Sample data (legacy)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🧠 Model Architecture

The VLA model consists of three main components:

1. **Vision Encoder**: A CNN that processes 64x64 RGB images
   - 2 convolutional layers with ReLU activation
   - Max pooling for dimensionality reduction
   - Outputs 256-dimensional feature vector

2. **Language Encoder**: Processes text instructions
   - Word embeddings for vocabulary representation
   - Linear layer to reduce to 256 dimensions
   - Averages word embeddings for sentence representation

3. **Fusion Network**: Combines vision and language features
   - Concatenates 256-dim vision + 256-dim language features
   - 2-layer MLP with ReLU activation
   - Outputs action probabilities (10 action classes, currently mock labels)

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd VLA
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install torchvision pillow
   ```

## 📥 Dataset: Flickr8k

This project now uses the [Flickr8k dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) for real vision-language training.

### Downloading Flickr8k
1. **Get your Kaggle API key** and place `kaggle.json` in `~/.kaggle/`.
2. **Download and extract the dataset**:
   ```bash
   mkdir -p data/flickr8k
   cd data/flickr8k
   kaggle datasets download -d adityajn105/flickr8k
   unzip -n flickr8k.zip
   cd ../..
   ```

### Preprocessing Flickr8k
Convert the captions to the project format:
```bash
python scripts/preprocess_flickr8k.py
```
This creates `data/flickr8k/flickr8k_data.json` for training.

## 📊 Data Format

The training data is in JSON format:
```json
[
    {
        "image": "data/flickr8k/Images/1000268201_693b08cb0e.jpg",
        "instruction": "A child in a pink dress is climbing up a set of stairs in an entry way ."
    },
    ...
]
```

## 🎯 Usage

### Training the Model on Flickr8k

To train the VLA model on real Flickr8k data:

```bash
python scripts/train.py
```

The training script will:
- Load training data from `data/flickr8k/flickr8k_data.json`
- Load and preprocess real images from disk
- Build vocabulary from instructions
- Initialize the VLA model
- Train for 5 epochs using cross-entropy loss (mock action labels)
- Save the trained model to `models/trained_vla_model.pth`

### Running Inference

To run inference with a trained model:

```bash
python scripts/run.py
```

The inference script will:
- Load the trained model
- Process a sample instruction
- Generate and output the predicted action

### Custom Usage

You can also use the model programmatically:

```python
import torch
from models.vla_model import VLAModel
from PIL import Image
import torchvision.transforms as transforms

# Initialize model
model = VLAModel(vocab_size=100, action_dim=10)
model.load_state_dict(torch.load('models/trained_vla_model.pth'))
model.eval()

# Prepare input
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
image = transform(Image.open('data/flickr8k/Images/1000268201_693b08cb0e.jpg').convert('RGB')).unsqueeze(0)
instruction = torch.LongTensor([[0, 1, 2]])  # Tokenized instruction

# Get action
with torch.no_grad():
    action = model(image, instruction)
    predicted_action = action.argmax().item()
```

## 🔧 Configuration

Key parameters can be modified in the training script:
- `NUM_EPOCHS`: Number of training epochs (default: 5)
- `LEARNING_RATE`: Learning rate for Adam optimizer (default: 0.001)
- `DATA_PATH`: Path to training data JSON file
- `MODEL_SAVE_PATH`: Path to save trained model

## 📈 Model Performance

The current implementation uses mock action labels (random integers). For production use, you would need:
- Real action labels or expert demonstrations
- Proper evaluation metrics
- Hyperparameter tuning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

[Add your license information here]

## 🙏 Acknowledgments

This project demonstrates the basic concepts of vision-language-action models. For more advanced implementations, consider exploring:
- CLIP (Contrastive Language-Image Pre-training)
- Vision Transformers
- Large Language Models for instruction following
- Reinforcement learning from human feedback (RLHF)

## 📝 Vision-Language Model Fine-Tuning (New!)

A new Jupyter notebook, `VLM_finetune.ipynb`, demonstrates how to fine-tune a large vision-language model (Qwen/Qwen2-VL-7B-Instruct) using LoRA adapters and the TRL library. This notebook covers:

- Loading and preparing datasets
- Setting up the Qwen2-VL-7B-Instruct model with LoRA adapters
- Training and evaluation using the TRL library
- Saving checkpoints and model cards to the `output/` directory

### Running the Notebook

1. Install the additional dependencies (if not already installed):
   ```bash
   pip install bitsandbytes peft trl transformers datasets
   ```
2. Open `VLM_finetune.ipynb` in Jupyter and run the cells step by step.

The notebook is self-contained and includes code for data formatting, model setup, training, and evaluation. Outputs and checkpoints will be saved in the `output/` directory. 