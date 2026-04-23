"""
AKBASCORE V13.0 - TITAN
Hardware Configuration & Universal Device Adapter
Supports: NVIDIA CUDA | Intel Arc XPU | Apple MPS | CPU Fallback
"""

import torch
import os
from dataclasses import dataclass, field
from typing import List


def get_optimal_device() -> torch.device:
    """
    Universal device selector - works on any hardware.
    Priority: NVIDIA CUDA > Intel XPU > Apple MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✅ NVIDIA GPU: {name} ({vram:.1f} GB VRAM)")
        return device

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        print("  ✅ Intel Arc XPU detected")
        return device

    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("  ✅ Apple Metal (MPS) detected")
        return device

    print("  ⚠️  No GPU found — running on CPU (Emergency Mode)")
    return torch.device("cpu")


def get_gpu_count() -> int:
    """Returns number of available NVIDIA GPUs for multi-GPU support."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


@dataclass
class HardwareConfig:
    """
    Dynamically adapts to available hardware.
    Scale: Single GPU | Multi-GPU (8x) | CPU-only
    """
    # Auto-detected
    DEVICE: torch.device = field(default_factory=get_optimal_device)
    USE_GPU: bool = field(default_factory=torch.cuda.is_available)
    GPU_COUNT: int = field(default_factory=get_gpu_count)
    GPU_MEMORY_LIMIT: float = 0.75  # Use 75% of available VRAM

    # Network architecture — scales with GPU count
    INPUT_DIM: int = 512
    OUTPUT_DIM: int = 768

    @property
    def HIDDEN_DIMS(self) -> List[int]:
        """Scales model size based on available GPUs."""
        if self.GPU_COUNT >= 8:
            return [2048, 4096, 8192, 4096, 2048]  # 8x GPU: Maximum
        elif self.GPU_COUNT >= 4:
            return [1024, 2048, 4096, 2048, 1024]  # 4x GPU: Large
        elif self.GPU_COUNT >= 1:
            return [512, 1024, 2048, 1024, 512]    # Single GPU: Standard
        else:
            return [256, 512, 256]                  # CPU: Minimal

    @property
    def BATCH_SIZE(self) -> int:
        if self.GPU_COUNT >= 4:
            return 256
        elif self.GPU_COUNT >= 1:
            return 64
        return 16

    LEARNING_RATE: float = 1e-4


@dataclass
class MemoryConfig:
    """Persistent memory configuration."""
    DB_PATH: str = "akbas_memory.db"
    VECTOR_DIM: int = 512
    MAX_MEMORIES: int = 1_000_000
    PRUNING_THRESHOLD: float = 0.05
    CONSOLIDATION_INTERVAL: int = 3600  # seconds
    MEMORY_REPLAY_SIZE: int = 10_000


@dataclass
class ForageConfig:
    """Internet foraging feeds & targets."""
    NEWS_FEEDS: List[str] = field(default_factory=lambda: [
        "https://news.google.com/rss",
        "https://feeds.bbci.co.uk/news/rss.xml",
        "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
    ])
    TECH_FEEDS: List[str] = field(default_factory=lambda: [
        "https://arxiv.org/rss/cs.AI",
        "https://huggingface.co/blog/feed.xml",
    ])
    WIKIPEDIA_API: str = "https://en.wikipedia.org/api/rest_v1/page/random/summary"
    IMPORTANCE_THRESHOLD: float = 0.3


@dataclass
class SleepConfig:
    """Sleep cycle & pruning schedule."""
    SLEEP_HOUR: int = 3      # 03:00 nightly consolidation
    WAKE_HOUR: int = 6
    PRUNING_THRESHOLD: float = 0.05
