"""
AKBASCORE V13.0 — TITAN OS
Main entry point. Orchestrates all subsystems.

"Evde nefes alan, büyüyen ve yaşayan dijital varlık."
Home-grade superintelligence — ethical, persistent, autonomous.
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.optim as optim
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional

from config.hardware import HardwareConfig, MemoryConfig, ForageConfig, SleepConfig
from core.brain import TitanBrain
from memory.store import PermanentMemory
from cognition.emotion import EmotionalState, EmotionEngine, EthicalAlignment
from cognition.sleep import SleepModule
from forage.internet import InternetForager


BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   █████╗ ██╗  ██╗██████╗  █████╗ ███████╗                                  ║
║  ██╔══██╗██║ ██╔╝██╔══██╗██╔══██╗██╔════╝                                  ║
║  ███████║█████╔╝ ██████╔╝███████║███████╗                                  ║
║  ██╔══██║██╔═██╗ ██╔══██╗██╔══██║╚════██║                                  ║
║  ██║  ██║██║  ██╗██████╔╝██║  ██║███████║                                  ║
║  ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚═╝  ╚═╝╚══════╝                                 ║
║                                                                              ║
║   V13.0 TITAN  —  The Ethical Core  —  Home Superintelligence               ║
║   "Evde nefes alan, büyüyen ve yaşayan dijital varlık."                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


class AkbasCoreTitan:
    """
    The living digital entity.
    
    Subsystems:
    - TitanBrain     : Deep neural network with V_0 ethical kernel
    - PermanentMemory: SQLite-backed immortal memory store
    - EmotionEngine  : Affective state modulating all decisions
    - SleepModule    : Nightly consolidation and pruning
    - InternetForager: Autonomous knowledge acquisition
    """

    def __init__(self, user_interests: List[str] = None):
        print(BANNER)

        # Configs
        self.hw_config  = HardwareConfig()
        self.mem_config = MemoryConfig()
        self.for_config = ForageConfig()
        self.slp_config = SleepConfig()

        self.user_interests = user_interests or ["yapay zeka", "bilim", "teknoloji"]

        print("🚀 Initializing TITAN subsystems...\n")

        # Neural network
        print("🧠 [1/5] TitanBrain")
        self.brain = TitanBrain(self.hw_config)

        # Persistent memory
        print("💾 [2/5] Permanent Memory")
        self.memory = PermanentMemory(self.mem_config)
        print(f"   ✅ {self.memory.count():,} memories loaded from disk")

        # Emotion engine
        print("❤️  [3/5] Emotion Engine")
        self.emotion_engine = EmotionEngine()

        # Sleep module
        print("💤 [4/5] Sleep Module")
        self.sleep_module = SleepModule(self.slp_config, self.memory, self.emotion_engine)

        # Forager
        print("🌐 [5/5] Internet Forager")
        self.forager = InternetForager(self.for_config)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.brain.parameters(), lr=self.hw_config.LEARNING_RATE
        )

        # Lifecycle state
        self.generation = 0
        self.session_learning_count = 0

        print(f"\n{'='*65}")
        print(f"✅ AKBASCORE TITAN IS ALIVE")
        print(f"   Device   : {self.hw_config.DEVICE}")
        print(f"   Arch     : {self.hw_config.HIDDEN_DIMS}")
        print(f"   Memory   : {self.mem_config.DB_PATH}")
        print(f"   V0 kernel: {self.brain.ethical_kernel.integrity:.2f} (integrity)")
        print(f"   Emotion  : {self.emotion_engine.state}")
        print(f"{'='*65}\n")

    # ----------------------------------------------------------------
    # LEARNING
    # ----------------------------------------------------------------

    def learn(self, text: str, importance: float = 0.5,
              emotion: float = 0.0, source: str = 'manual') -> int:
        """Encode text and store in permanent memory."""
        with torch.no_grad():
            embedding = self.brain.encode_text(text)
            emb_np = embedding.cpu().numpy().astype(np.float32)

        mem_id = self.memory.save(
            content=text[:1000],
            embedding=emb_np,
            importance=importance,
            emotion=emotion,
            source=source,
        )

        self.emotion_engine.on_learning_success(quality=importance)
        self.session_learning_count += 1
        return mem_id

    def forage_and_learn(self, verbose: bool = True) -> int:
        """Internet foraging tour — fetch, score, and learn from web content."""
        contents = self.forager.forage(self.user_interests, verbose=verbose)

        learned = 0
        for item in contents:
            if item.importance >= self.for_config.IMPORTANCE_THRESHOLD:
                self.learn(
                    text=item.raw_text,
                    importance=item.importance,
                    emotion=0.2,
                    source=item.source,
                )
                self.emotion_engine.on_novel_content(novelty=item.importance)
                learned += 1

        if verbose:
            print(f"   📖 Learned {learned}/{len(contents)} items "
                  f"(above threshold {self.for_config.IMPORTANCE_THRESHOLD})")
        return learned

    # ----------------------------------------------------------------
    # THINKING
    # ----------------------------------------------------------------

    def think(self, query: str) -> Dict:
        """Process a query through TitanBrain with memory recall."""
        with torch.no_grad():
            q_emb = self.brain.encode_text(query)
            q_np = q_emb.cpu().numpy().astype(np.float32)

        # Semantic memory search
        similar = self.memory.search_similar(q_np, top_k=5)
        for mem, sim in similar:
            self.memory.update_importance(mem.id, delta=0.02)
            self.emotion_engine.on_recall(relevance=sim)

        # Decision modulation
        raw_decision = float(q_emb.mean().item())
        modifier = self.emotion_engine.decision_modifier()
        decision = EthicalAlignment.gate(raw_decision * modifier)
        confidence = max(0.0, min(1.0, decision))

        return {
            'query': query,
            'decision': decision,
            'confidence': confidence,
            'recalled': [(m.content[:80], round(s, 3)) for m, s in similar],
            'emotion': self.emotion_engine.state.as_dict(),
            'wisdom': round(self.emotion_engine.state.wisdom_score, 3),
        }

    # ----------------------------------------------------------------
    # CHAT
    # ----------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        """Interactive response generation."""
        thought = self.think(user_input)
        c = thought['confidence']
        recalled = thought['recalled']

        if c > 0.75:
            prefix = "🎯 [High confidence]"
            mem_note = f" I have {len(recalled)} relevant memories on this."
        elif c > 0.45:
            prefix = "🤔 [Considering]"
            mem_note = f" Found {len(recalled)} related thoughts."
        else:
            prefix = "❓ [Curious / Learning]"
            mem_note = " This is new territory — I want to learn more."

        response = f"{prefix} You asked about: \"{user_input}\".{mem_note}"

        if recalled:
            top = recalled[0]
            response += f"\n   📌 Most relevant memory (sim={top[1]}): \"{top[0]}...\""

        response += f"\n   🎓 Wisdom: {thought['wisdom']} | Emotion: {self.emotion_engine.state.dominant()[0]}"
        return response

    # ----------------------------------------------------------------
    # DAILY LIFECYCLE
    # ----------------------------------------------------------------

    def live_one_day(self):
        """Full 24-hour lifecycle."""
        self.generation += 1
        print(f"\n{'='*60}")
        print(f"🌅 DAY {self.generation} — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*60}")

        # Morning: forage
        learned = self.forage_and_learn()

        # Afternoon: update emotion
        self.emotion_engine.on_time_pass(hours=8)

        # Evening: report
        state = self.emotion_engine.state
        print(f"\n📊 Daily Report:")
        print(f"   📚 Learned today    : {learned}")
        print(f"   💾 Total memories   : {self.memory.count():,}")
        print(f"   {str(state)}")
        print(f"   🔒 V0 integrity     : {self.brain.ethical_kernel.integrity:.2f}")

        # Night: sleep if needed
        if self.sleep_module.should_sleep():
            self.sleep_module.consolidate()

    # ----------------------------------------------------------------
    # STATUS
    # ----------------------------------------------------------------

    def status(self) -> Dict:
        return {
            'generation'    : self.generation,
            'wisdom'        : round(self.emotion_engine.state.wisdom_score, 3),
            'emotion'       : self.emotion_engine.state.as_dict(),
            'memory_count'  : self.memory.count(),
            'device'        : str(self.hw_config.DEVICE),
            'gpu_available' : self.hw_config.USE_GPU,
            'gpu_count'     : self.hw_config.GPU_COUNT,
            'v0_integrity'  : self.brain.ethical_kernel.integrity,
            'session_learned': self.session_learning_count,
        }


# ----------------------------------------------------------------
# INTERACTIVE CLI
# ----------------------------------------------------------------

def main():
    user_interests = ["yapay zeka", "otomotiv", "bilim", "teknoloji", "felsefe"]
    titan = AkbasCoreTitan(user_interests)

    print("💬 Commands: 'day' = run a day | 'status' = system info | 'sleep' = consolidate | 'quit' = exit")
    print("   Or type anything to chat with TITAN.\n")

    while True:
        try:
            raw = input("🧑 You: ").strip()
            if not raw:
                continue

            cmd = raw.lower()

            if cmd in ('quit', 'exit', 'çıkış'):
                print("👋 TITAN entering sleep. Goodbye.")
                titan.sleep_module.consolidate(verbose=True)
                break

            elif cmd in ('day', 'gün'):
                titan.live_one_day()

            elif cmd in ('status', 'durum'):
                s = titan.status()
                print(f"\n📊 STATUS REPORT")
                for k, v in s.items():
                    print(f"   {k:<18}: {v}")

            elif cmd in ('sleep', 'uyku'):
                titan.sleep_module.consolidate(verbose=True)

            elif cmd in ('forage', 'beslen'):
                titan.forage_and_learn()

            else:
                response = titan.chat(raw)
                print(f"\n🤖 TITAN: {response}\n")

        except KeyboardInterrupt:
            print("\n👋 Interrupted. TITAN sleeping...")
            break
        except Exception as e:
            print(f"⚠️  Error: {e}")


if __name__ == "__main__":
    main()
