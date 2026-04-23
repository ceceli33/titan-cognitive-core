"""
AKBASCORE V13.0 — Sleep & Consolidation
Nightly memory pruning, importance strengthening, and dream replay.

Inspired by hippocampal consolidation in biological brains:
during sleep, the brain replays experiences and strengthens
important pathways while discarding noise.
"""

import random
import time
from datetime import datetime
from typing import TYPE_CHECKING

from config.hardware import SleepConfig

if TYPE_CHECKING:
    from memory.store import PermanentMemory
    from cognition.emotion import EmotionEngine


class SleepModule:
    """
    Nightly consolidation cycle.
    
    Phase 1 — Pruning:    Delete memories below importance threshold
    Phase 2 — Replay:     Re-process top memories (simulated dream)
    Phase 3 — Strengthen: Boost importance of frequently accessed memories
    Phase 4 — Restore:    Reset emotional fatigue
    """

    def __init__(self, config: SleepConfig, memory: "PermanentMemory",
                 emotion_engine: "EmotionEngine"):
        self.config = config
        self.memory = memory
        self.emotion = emotion_engine
        self.dream_count = 0
        self.last_sleep: datetime = None

    def should_sleep(self) -> bool:
        hour = datetime.now().hour
        return hour >= self.config.SLEEP_HOUR or hour < self.config.WAKE_HOUR

    def consolidate(self, verbose: bool = True) -> dict:
        """
        Run full consolidation cycle. Returns summary report.
        """
        start = time.time()
        if verbose:
            print("\n💤 Sleep cycle initiated...")
            print(f"   🕐 Time: {datetime.now().strftime('%H:%M:%S')}")

        # Phase 1: Prune weak memories
        pruned = self.memory.prune_weak(self.config.PRUNING_THRESHOLD)
        overflow_pruned = self.memory.prune_overflow()
        if verbose:
            print(f"   ✂️  Pruned: {pruned + overflow_pruned} weak memories")

        # Phase 2: Identify important memories
        top_memories = self.memory.recall_top(100)
        if verbose:
            print(f"   ⭐ Strengthening {len(top_memories)} important memories")

        # Phase 3: Strengthen frequently accessed memories
        strengthened = 0
        for mem in top_memories:
            if mem.access_count > 2:
                self.memory.update_importance(mem.id, delta=0.03)
                strengthened += 1

        # Phase 4: Dream replay
        self.dream_count += 1
        replayed = self._dream_replay(top_memories, batch_size=32)
        if verbose:
            print(f"   💭 Dream #{self.dream_count}: {replayed} memories replayed")

        # Phase 5: Emotional restoration
        self.emotion.on_sleep()
        if verbose:
            print(f"   🌅 Woke up — emotional state: {self.emotion.state}")

        self.last_sleep = datetime.now()
        elapsed = time.time() - start

        report = {
            'pruned': pruned + overflow_pruned,
            'strengthened': strengthened,
            'replayed': replayed,
            'dream_number': self.dream_count,
            'duration_seconds': round(elapsed, 2),
            'memory_count': self.memory.count(),
        }

        if verbose:
            print(f"   ✅ Consolidation complete in {elapsed:.1f}s")

        return report

    def _dream_replay(self, memories, batch_size: int = 32) -> int:
        """
        Simulate dream state: randomly sample memories and 'replay' them.
        In a full implementation, these would be re-fed through the network.
        """
        if not memories:
            return 0

        sample_size = min(batch_size, len(memories))
        selected = random.sample(memories, sample_size)

        for mem in selected:
            # Mark as accessed (strengthens recall pathway)
            self.memory.update_importance(mem.id, delta=0.01)

        return sample_size
