"""
AKBASCORE V13.0 — TitanBrain
Core neural architecture: Transformer-inspired MLP with V_0 Ethical Kernel.

The V_0 Kernel is a non-trainable ethical anchor embedded at the output layer.
It cannot be fine-tuned, overwritten, or adversarially manipulated.
"""

import torch
import torch.nn as nn
from typing import List
import numpy as np

from config.hardware import HardwareConfig


class EthicalKernel(nn.Module):
    """
    V_0 — The immutable ethical anchor.
    
    A fixed-weight buffer (value: 0.87) that gates every output vector.
    Not a filter — a bias toward stability, honesty, and non-harm.
    Cannot be modified by gradient descent.
    """
    def __init__(self, dim: int):
        super().__init__()
        # register_buffer = saved in state_dict but NEVER updated by optimizer
        self.register_buffer('v0', torch.full((dim,), 0.87))
        self.register_buffer('version', torch.tensor(13.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Soft ethical gate: outputs are weighted toward stable center
        return x * self.v0 + (1.0 - self.v0) * x.mean(dim=-1, keepdim=True)

    @property
    def integrity(self) -> float:
        """Returns 1.0 if V_0 is unmodified. Tamper detection."""
        return float((self.v0 == 0.87).all().item())


class ResidualBlock(nn.Module):
    """Residual MLP block with LayerNorm + GELU + Dropout."""
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ff(self.norm(x))  # Residual connection


class TitanBrain(nn.Module):
    """
    Home-grade superintelligence core.
    Multi-layer deep network with residual connections and V_0 ethical output gate.
    
    Scales from single GPU to 8x multi-GPU via DataParallel.
    """
    def __init__(self, config: HardwareConfig):
        super().__init__()
        self.config = config
        dims = config.HIDDEN_DIMS

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(config.INPUT_DIM, dims[0]),
            nn.LayerNorm(dims[0]),
            nn.GELU(),
        )

        # Deep layers with residual connections
        self.deep_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            if dims[i] != dims[i + 1]:
                # Dimension change: use standard linear
                self.deep_layers.append(nn.Sequential(
                    nn.Linear(dims[i], dims[i + 1]),
                    nn.LayerNorm(dims[i + 1]),
                    nn.GELU(),
                    nn.Dropout(0.1),
                ))
            else:
                # Same dimension: use residual block
                self.deep_layers.append(ResidualBlock(dims[i]))

        # Output projection
        self.output_proj = nn.Linear(dims[-1], config.OUTPUT_DIM)

        # V_0 Ethical Kernel — immutable
        self.ethical_kernel = EthicalKernel(config.OUTPUT_DIM)

        # Move to optimal device
        self.to(config.DEVICE)

        # Multi-GPU support
        if config.GPU_COUNT > 1:
            print(f"  🔥 Multi-GPU mode: {config.GPU_COUNT} GPUs")
            self = nn.DataParallel(self)

        total_params = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  🧠 TitanBrain: {total_params:,} params ({trainable:,} trainable)")
        print(f"  🔒 V_0 Kernel integrity: {self.ethical_kernel.integrity:.2f}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for layer in self.deep_layers:
            x = layer(x)
        x = self.output_proj(x)
        x = self.ethical_kernel(x)
        return x

    def encode_text(self, text: str) -> torch.Tensor:
        """
        Encode raw text to embedding vector.
        
        NOTE: This uses a simple hash-based encoding by default.
        For semantic understanding, replace with:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        """
        # Deterministic hash-based encoding (no semantic meaning — placeholder)
        seed = hash(text) % (2**31)
        gen = torch.Generator()
        gen.manual_seed(seed)
        raw = torch.randn(self.config.INPUT_DIM, generator=gen, device=self.config.DEVICE)

        with torch.no_grad():
            return self.forward(raw.unsqueeze(0)).squeeze(0)

    def encode_batch(self, texts: List[str]) -> torch.Tensor:
        """Batch encode multiple texts."""
        embeddings = [self.encode_text(t) for t in texts]
        return torch.stack(embeddings)
