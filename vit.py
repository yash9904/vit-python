import logging
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

np.random.seed(0)
torch.manual_seed(0)


def get_logger(name, level=logging.INFO):
    """
    Get a logger with the given name and level.

    Args:
        name (str): The name of the logger.
        level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: The logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logger.datefmt = "%Y-%m-%d %H:%M"

    return logger


logger = get_logger(__name__)


def patch_embedding(
    images: Union[np.ndarray, torch.Tensor],
    n_patches_w: int,
    n_patches_h: int,
):
    n, c, h, w = images.shape

    assert h == w, "The height of the image should be equal to its width"
    assert (
        h % n_patches_h == 0 and w % n_patches_w == 0
    ), "The number of patches should be a divisor of the image size in both dimensions"
    patches = torch.zeros(
        (n, n_patches_h * n_patches_w, h * w * c // (n_patches_h * n_patches_w))
    ).to(images.device)
    patch_size_h = h // n_patches_h
    patch_size_w = w // n_patches_w

    for i in range(n):
        img = images[i]
        for ph in range(n_patches_h):
            for pw in range(n_patches_w):
                patch = img[
                    :,
                    ph * patch_size_h : (ph + 1) * patch_size_h,
                    pw * patch_size_w : (pw + 1) * patch_size_w,
                ]
                patches[i, ph * n_patches_h + pw] = patch.reshape(-1)
    # convert to torch float32
    patches: torch.Tensor = patches.float()
    return patches


def positional_embedding(sequence_length: int, dim: int) -> torch.Tensor:
    pos_embedding = torch.ones(sequence_length, dim)
    for i in range(sequence_length):
        for j in range(dim):
            if j % 2 == 0:
                pos_embedding[i, j] = np.sin(i / 10000 ** (j / dim))
            else:
                pos_embedding[i, j] = np.cos(i / 10000 ** ((j - 1) / dim))

    return pos_embedding


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d = d_model
        self.n_heads = n_heads

        assert d_model % n_heads == 0, "d_model should be divisible by n_heads"
        self.d_head = d_model // n_heads
        self.q = nn.ModuleList(
            [nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)]
        )
        self.k = nn.ModuleList(
            [nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)]
        )
        self.v = nn.ModuleList(
            [nn.Linear(self.d_head, self.d_head) for _ in range(self.n_heads)]
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences, mask=None):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):

                q_mapping = self.q[head]
                k_mapping = self.k[head]
                v_mapping = self.v[head]

                seq = sequence[:, head * self.d_head : (head + 1) * self.d_head]
                q = q_mapping(seq)
                k = k_mapping(seq)
                v = v_mapping(seq)

                attention = self.softmax(
                    torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.d_head)
                )
                if mask is not None:
                    attention = attention * mask
                seq_result.append(torch.matmul(attention, v))

            result.append(torch.cat(seq_result, dim=-1))
        return torch.stack(result, dim=0)


class EncoderVIT(nn.Module):
    def __init__(self, hidden_dim, n_heads, mlp_ratio=4):
        super(EncoderVIT, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.mhsa = MultiHeadAttention(self.hidden_dim, self.n_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x):
        x = x + self.mhsa(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VIT(nn.Module):
    def __init__(
        self,
        shape=(1, 28, 28),
        n_patches_w=7,
        n_patches_h=7,
        hidden_dim=8,
        out_dim=10,
        n_blocks=2,
        n_heads=2,
        encoder_mlp_ratio=4,
    ):
        super(VIT, self).__init__()
        self.input_shape = shape
        self.hidden_dim = hidden_dim
        self.n_patches_w = n_patches_w
        self.n_patches_h = n_patches_h
        self.out_dim = out_dim
        self.n_heads = n_heads
        self.n_blocks = n_blocks
        self.encoder_mlp_ratio = encoder_mlp_ratio
        assert (self.input_shape[1] % self.n_patches_h == 0) and (
            self.input_shape[2] % self.n_patches_w == 0
        ), "The number of patches should be a divisor of the image size in both dimensions"
        self.patch_size = (shape[1] // n_patches_h, shape[2] // n_patches_w)
        self.class_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.input_dim = int(shape[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapping = nn.Linear(self.input_dim, self.hidden_dim)
        self.positional_embedding = nn.Parameter(
            torch.tensor(
                positional_embedding((n_patches_w * n_patches_h) + 1, hidden_dim),
            ),
            requires_grad=False,
        )
        self.blocks = nn.ModuleList(
            [
                EncoderVIT(
                    self.hidden_dim,
                    n_heads=self.n_heads,
                    mlp_ratio=self.encoder_mlp_ratio,
                )
                for _ in range(self.n_blocks)
            ]
        )
        self.mlp_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.out_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        logger.info(f"Input shape: {x.shape}")
        patches = patch_embedding(x, self.n_patches_w, self.n_patches_h)
        logger.info(f"Patches shape: {patches.shape}")
        tokens = self.linear_mapping(patches)
        logger.info(f"Tokens shape: {tokens.shape}")
        tokens = torch.cat(
            (self.class_embedding.expand(x.shape[0], -1, -1), tokens), dim=1
        )
        logger.info(f"Tokens shape after class embedding: {tokens.shape}")
        out = tokens + self.positional_embedding
        logger.info(f"Tokens shape after positional embedding: {out.shape}")
        for block in self.blocks:
            out = block(out)
        logger.info(f"Tokens shape after blocks: {out.shape}")
        out = out[:, 0]
        logger.info(f"Tokens shape after slicing: {out.shape}")
        out = self.mlp_classifier(out)
        logger.info(f"Output shape: {out.shape}")

        return out
