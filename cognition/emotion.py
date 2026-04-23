"""
AKBASCORE V13.0 — Cognition & Emotional Intelligence
Emotional state engine + V_0 ethical alignment layer.

Emotions are not decorative. They modulate decision strength,
learning priority, and the depth of memory consolidation.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple
import math


@dataclass
class EmotionalState:
    """
    Internal affective state of AkbasCore.
    All values are clamped to [0.0, 1.0].
    """
    curiosity:    float = 0.70
    satisfaction: float = 0.30
    anxiety:      float = 0.20
    fatigue:      float = 0.10
    wonder:       float = 0.50

    def clamp(self):
        for attr in ['curiosity', 'satisfaction', 'anxiety', 'fatigue', 'wonder']:
            setattr(self, attr, max(0.0, min(1.0, getattr(self, attr))))

    def as_dict(self) -> Dict[str, float]:
        return {
            'curiosity':    round(self.curiosity, 3),
            'satisfaction': round(self.satisfaction, 3),
            'anxiety':      round(self.anxiety, 3),
            'fatigue':      round(self.fatigue, 3),
            'wonder':       round(self.wonder, 3),
        }

    def dominant(self) -> Tuple[str, float]:
        d = self.as_dict()
        return max(d.items(), key=lambda x: x[1])

    @property
    def wisdom_score(self) -> float:
        """
        Composite score: high satisfaction + low anxiety + high curiosity.
        Range: 0.0 → 1.0
        """
        return (
            self.satisfaction * 0.35 +
            (1.0 - self.anxiety) * 0.30 +
            self.curiosity * 0.25 +
            self.wonder * 0.10
        )

    def __str__(self) -> str:
        name, val = self.dominant()
        return f"{name.upper()} ({val:.2f}) | wisdom={self.wisdom_score:.3f}"


class EmotionEngine:
    """
    Updates emotional state based on events:
    - Learning success / failure
    - Novel information encountered
    - Sleep / rest cycles
    - Time passage (fatigue accumulation)
    """

    def __init__(self, state: EmotionalState = None):
        self.state = state or EmotionalState()

    def on_learning_success(self, quality: float = 0.5):
        """Called after successfully learning new content."""
        self.state.satisfaction += quality * 0.12
        self.state.anxiety      -= quality * 0.08
        self.state.fatigue      += 0.02  # Learning is tiring
        self.state.clamp()

    def on_novel_content(self, novelty: float = 0.5):
        """Called when encountering genuinely new information."""
        self.state.curiosity += novelty * 0.15
        self.state.wonder    += novelty * 0.10
        self.state.fatigue   += 0.01
        self.state.clamp()

    def on_recall(self, relevance: float = 0.5):
        """Called when a memory is successfully recalled."""
        self.state.satisfaction += relevance * 0.05
        self.state.clamp()

    def on_failure(self, severity: float = 0.3):
        """Called when something goes wrong (network error, parse fail, etc.)."""
        self.state.anxiety   += severity * 0.15
        self.state.curiosity -= severity * 0.05
        self.state.clamp()

    def on_sleep(self):
        """Deep restoration during consolidation cycle."""
        self.state.fatigue      = max(0.0, self.state.fatigue - 0.7)
        self.state.anxiety      = max(0.1, self.state.anxiety - 0.3)
        self.state.satisfaction = max(0.0, self.state.satisfaction - 0.1)  # Reset baseline
        self.state.clamp()

    def on_time_pass(self, hours: float = 1.0):
        """Passive fatigue accumulation."""
        self.state.fatigue += hours * 0.015
        self.state.clamp()

    def decision_modifier(self) -> float:
        """
        Returns a scalar [0.3, 1.2] that modulates decision confidence.
        High curiosity + low anxiety = stronger, bolder decisions.
        High fatigue = conservative, cautious decisions.
        """
        base = 0.5
        base += self.state.curiosity * 0.30
        base -= self.state.anxiety   * 0.20
        base -= self.state.fatigue   * 0.15
        base += self.state.wonder    * 0.10
        return max(0.3, min(1.2, base))

    def should_rest(self) -> bool:
        """True when fatigue exceeds healthy threshold."""
        return self.state.fatigue > 0.75


class EthicalAlignment:
    """
    V_0 alignment checks.
    Soft guardrails derived from the 0.87 ethical kernel constant.
    These are heuristic — not a substitute for real safety systems.
    """
    V0_CONSTANT = 0.87

    @classmethod
    def alignment_score(cls, decision_strength: float) -> float:
        """Returns how closely a decision aligns with V_0 principles."""
        # Perfect alignment = V0_CONSTANT, deviation in either direction lowers score
        return 1.0 - abs(decision_strength - cls.V0_CONSTANT)

    @classmethod
    def is_aligned(cls, decision_strength: float, threshold: float = 0.5) -> bool:
        return cls.alignment_score(decision_strength) >= threshold

    @classmethod
    def gate(cls, value: float) -> float:
        """Apply V_0 soft gate to any scalar output."""
        return value * cls.V0_CONSTANT + (1.0 - cls.V0_CONSTANT) * 0.5
