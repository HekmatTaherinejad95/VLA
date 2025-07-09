import torch
import torch.nn as nn

class VLAModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, action_dim=10):
        super(VLAModel, self).__init__()
        
        # Vision processing part
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 256) # Assuming 64x64 input images
        )
        
        # Language processing part
        self.language_encoder = nn.Embedding(vocab_size, embedding_dim)
        self.language_reducer = nn.Linear(embedding_dim, 256)

        # Combined model
        self.fusion = nn.Sequential(
            nn.Linear(256 + 256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, image, text):
        vision_features = self.vision_encoder(image)
        
        # For simplicity, we'll average the word embeddings
        language_features = self.language_encoder(text).mean(dim=1)
        language_features = self.language_reducer(language_features)
        
        combined_features = torch.cat([vision_features, language_features], dim=1)
        
        action = self.fusion(combined_features)
        return action
