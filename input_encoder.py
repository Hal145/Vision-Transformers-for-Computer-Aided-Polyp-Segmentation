import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, input_shape, patch_size, embedding_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim

        self.num_patches = (input_shape[1] // patch_size) * (input_shape[2] // patch_size)
        self.proj = nn.Linear(input_shape[0] * patch_size * patch_size, embedding_dim)
        self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))

    def forward(self, x):
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.reshape(x.size(0), x.size(1), x.size(2), x.size(3), -1)
        x = x.permute(0, 2, 3, 1, 4).reshape(x.size(0), x.size(2), x.size(3), -1)
        x = self.proj(x)
        x = x + self.positional_embedding[:, :x.size(1), :]

        return x
