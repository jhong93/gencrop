import math
import random
import secrets
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# Seed workers with real SystemRandom random to avoid torch silliness
# NOTE: this will make the dataloaders non-deterministic
def sysrand_init_worker(id):
    seed = secrets.randbits(32)
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


class SquarePad:

    def __call__(self, img):
        _, h, w = img.shape
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)

        eh, ev = 0, 0
        if h + 2 * vp < w + 2 * hp:
            ev = 1
        elif w + 2 * hp < h + 2 * vp:
            eh = 1
        padding = (hp, vp, hp + eh, vp + ev)
        return transforms.functional.pad(img, padding, 0, 'constant')
