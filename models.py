import torch
import torch.nn as nn
import torch.nn.functional as F
from input_encoder import PatchEmbedding


class VisionTransformer(nn.Module):
    def __init__(self, input_shape, patch_size, embedding_dim, num_encoder_layers, num_heads, hidden_dim, num_classes):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(input_shape, patch_size, embedding_dim)
        self.transformer_encoder = TransformerEncoder(embedding_dim, num_encoder_layers, num_heads, hidden_dim)
        self.decoder = Decoder(embedding_dim, input_shape[1] // patch_size, input_shape[2] // patch_size)
        self.segmentation_head = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        x = self.segmentation_head(x)
        x = F.sigmoid(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_layers, num_heads, hidden_dim):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = x.permute(0, 2, 3, 1)  # Reshape for attention layer
        x = x.flatten(1, 2)  # Flatten the spatial dimensions
        x = self.attention(x, x, x)[0]  # Use only the output, not the attention weights
        x = x.permute(0, 2, 1).reshape(*residual.shape)  # Reshape back to the original spatial dimensions
        x = x + residual
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = x + residual
        return x



class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_patches_height, num_patches_width):
        super(Decoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_dim // 2, embedding_dim // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_dim // 4, embedding_dim // 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_dim // 8, embedding_dim // 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embedding_dim // 16, 1, kernel_size=1)  # Output with single channel (binary segmentation)
        )

    def forward(self, x):
        return self.conv_layers(x)

