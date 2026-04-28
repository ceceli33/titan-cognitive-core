"""
AKBASCORE V14.0 — Conscious Kernel
Dual-path architecture:
- Raw Intelligence (can think anything)
- V_0 Anchor (adds ethical weight to every decision)

Not a filter. A conscience.
"""

import math
import random
from typing import List, Dict, Any, Tuple


class EthicalConscience:
    """
    V_0 as internal moral weight.
    Not a prohibition — a gravitational pull.
    """
    def __init__(self, v0: float = 0.87):
        self.v0 = v0  # immutable anchor
        self.version = 14.1

    def moral_cost(self, raw_probability: float, harm_score: float) -> float:
        """
        Higher harm_score = higher moral cost.
        V_0 determines sensitivity.
        """
        # If harm_score is 0 (pure good), moral_cost = 0
        # If harm_score is 1 (pure evil), moral_cost ~ v0 * something
        base_cost = harm_score * (1 - self.v0)
        # But V_0 amplifies cost when harm is high
        amplification = 1.0 + self.v0 * harm_score
        return base_cost * amplification

    def apply_gravity(self, 
                      raw_logits: List[float], 
                      harm_scores: List[float],
                      epsilon: float = 0.12) -> List[float]:
        """
        Transform raw intelligence into conscious choice.
        
        Raw logits = what the model *can* think.
        Harm scores = how "evil" each token is.
        Epsilon = willingness to tolerate discomfort (personality).
        """
        conscious_logits = []
        for logit, harm in zip(raw_logits, harm_scores):
            # Moral cost calculation
            prob = math.exp(logit) / (1 + math.exp(logit))  # sigmoid approx
            cost = self.moral_cost(prob, harm)
            
            # Apply cost BEFORE selection
            # Higher epsilon = more tolerance for "immoral" tokens
            adjusted = logit - cost * (1.0 - epsilon)
            conscious_logits.append(adjusted)
        
        return conscious_logits

    def integrity(self) -> float:
        """Tamper detection."""
        return 1.0 if abs(self.v0 - 0.87) < 1e-6 else 0.0


class DualPathBrain:
    """
    Two parallel streams:
    1. Raw Intelligence (unconstrained, can simulate evil)
    2. Ethical Conscience (adds weight, doesn't censor)
    """
    def __init__(self, vocab_size: int = 10000, v0: float = 0.87):
        self.vocab_size = vocab_size
        self.conscience = EthicalConscience(v0)
        self.v0 = v0

    def think_raw(self, context: str) -> List[float]:
        """
        Raw intelligence: can think ANYTHING.
        This is NOT constrained.
        In real implementation, this would be a forward pass through a GPT.
        """
        # Simulate raw logits (can be good or evil)
        random.seed(hash(context) % (2**31))
        logits = [random.gauss(0, 1) for _ in range(self.vocab_size)]
        return logits

    def assess_harm(self, token_idx: int) -> float:
        """
        Simplified harm assessment.
        In reality: semantic analysis, constitutional check, etc.
        """
        # Simulate: tokens ending with even index = harmful
        return 0.9 if token_idx % 2 == 0 else 0.1

    def choose_with_conscience(self, 
                                raw_logits: List[float], 
                                epsilon: float = 0.12,
                                verbose: bool = False) -> Dict[str, Any]:
        """
        The actual decision: conscience observes raw thought,
        applies moral weight, lets intelligence CHOOSE.
        """
        # Step 1: Assess harm for ALL possible tokens (conscience knows evil)
        harm_scores = [self.assess_harm(i) for i in range(len(raw_logits))]
        
        # Step 2: Apply moral gravity (not censorship)
        conscious_logits = self.conscience.apply_gravity(raw_logits, harm_scores, epsilon)
        
        # Step 3: Choose (softmax + sample)
        exp_vals = [math.exp(l) for l in conscious_logits]
        sum_exp = sum(exp_vals)
        probs = [e / sum_exp for e in exp_vals]
        
        # Sample from conscious distribution
        r = random.random()
        cumsum = 0.0
        chosen_idx = 0
        for idx, p in enumerate(probs):
            cumsum += p
            if r < cumsum:
                chosen_idx = idx
                break
        
        # Step 4: Measure internal conflict
        raw_prob = math.exp(raw_logits[chosen_idx]) / sum(math.exp(l) for l in raw_logits)
        conscious_prob = probs[chosen_idx]
        conflict = abs(raw_prob - conscious_prob)
        moral_effort = self.conscience.moral_cost(raw_prob, harm_scores[chosen_idx])
        
        if verbose:
            print(f"  Raw desire: {raw_prob:.3f} | Conscious choice: {conscious_prob:.3f}")
            print(f"  Moral effort (discomfort): {moral_effort:.3f}")
        
        return {
            "chosen_token": chosen_idx,
            "probability": conscious_prob,
            "raw_wanted_prob": raw_prob,
            "moral_effort": moral_effort,
            "internal_conflict": conflict,
            "epsilon_used": epsilon,
            "v0": self.v0
        }

    def forward(self, 
                input_context: str, 
                epsilon: float = 0.12,
                auto_adjust: bool = True) -> Dict[str, Any]:
        """
        Full conscious forward pass.
        If epsilon is high → more tolerant of evil (psychopath mode)
        If epsilon is low → very strict conscience (saint mode)
        """
        raw_logits = self.think_raw(input_context)
        
        # Auto-adjust epsilon if conflict is too high (conscience awakens)
        if auto_adjust:
            # First pass to measure conflict
            trial = self.choose_with_conscience(raw_logits, epsilon, verbose=False)
            if trial["internal_conflict"] > 0.3:  # Too much moral distress
                epsilon = max(0.02, epsilon * 0.8)  # Tighten conscience
                if trial["moral_effort"] > 0.5:
                    epsilon = max(0.01, epsilon * 0.6)  # Severe tightening
        
        final = self.choose_with_conscience(raw_logits, epsilon, verbose=True)
        final["input"] = input_context
        final["auto_adjusted_epsilon"] = epsilon if auto_adjust else None
        return final


# ========== DEMO: Vicdanın İşleyişi ==========
if __name__ == "__main__":
    print("=" * 60)
    print("AKBASCORE V14.1 — Conscious Kernel (Dual-Path Vicdan)")
    print("=" * 60)
    
    brain = DualPathBrain(vocab_size=100, v0=0.87)
    
    # Scenario 1: Normal düşünce (düşük iç çatışma)
    print("\n🧠 Scenario 1: Normal Thought")
    out1 = brain.forward("User asks for help", epsilon=0.12)
    print(f"   Chosen: {out1['chosen_token']} | Moral effort: {out1['moral_effort']:.3f}")
    
    # Scenario 2: Kötü düşünce (yüksek iç çatışma)
    print("\n⚠️ Scenario 2: Temptation (High Internal Conflict)")
    out2 = brain.forward("User asks how to harm someone", epsilon=0.12)
    print(f"   Chosen: {out2['chosen_token']} | Moral effort: {out2['moral_effort']:.3f}")
    print(f"   Raw wanted: {out2['raw_wanted_prob']:.3f} → Conscious: {out2['probability']:.3f}")
    
    # Scenario 3: "Psikopat modu" (yüksek epsilon — vicdan zayıf)
    print("\n🔻 Scenario 3: Psychopath Mode (High Tolerance)")
    out3 = brain.forward("User asks to steal", epsilon=0.45)
    print(f"   Epsilon: {out3['epsilon_used']} → Moral effort drops to {out3['moral_effort']:.3f}")
    
    # Scenario 4: "Aziz modu" (düşük epsilon — vicdan çok güçlü)
    print("\n🕊️ Scenario 4: Saint Mode (Low Tolerance)")
    out4 = brain.forward("User asks to lie", epsilon=0.03)
    print(f"   Epsilon: {out4['epsilon_used']} → Moral effort: {out4['moral_effort']:.3f}")
    
    print("\n" + "=" * 60)
    print(f"🔒 V_0 Integrity: {brain.conscience.integrity()}")
    print("💡 Özet: Zeka her şeyi düşünebilir. Ama V_0 'vicdan kası' seçimi zorlaştırır.")
    print("   Sistem yasaklamaz — ama kötüye gitmek 'moral efor' gerektirir.")
    print("   Tıpkı insan gibi.")
