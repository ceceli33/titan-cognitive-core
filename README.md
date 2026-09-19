[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22735745.svg)](https://doi.org/10.5281/zenodo.22735745)


AkbasCore is a sub-threshold, inference-layer steering framework for transformer models. It employs Damped Resonance Alignment (DRA) to mathematically adjust a model's hidden states during the forward pass—enabling precise, modular alignment without fine-tuning, weight modification, or prompt engineering. This repository is the canonical record for the DRA methodology, the Pusula compass-construction technique, the Phase V synthetic concept-injection engine, and all associated original terminology.

# AkbasCore — Damped Resonance Alignment (DRA)
A sub-threshold, inference-layer activation steering framework for transformer language models.

Author: Akbaş Status: Independent research pre-print — not yet peer-reviewed Version: 2.1 — DRA / SYNTHETIC INJECTION (September 2026) · Phase IV product line (AkbasCore 1.3 DRA — ethical Pusula, Qwen2.5-1.5B, CPU) remains documented below for lineage and is still maintained · the Phase V 2.0 multi-axis Manifold branch is retained as a documented exploratory branch with limited yield · Phase V 2.1 is the current active line · baseline 1.2 (June 2026) remains documented below for lineage License: MIT (kernel code) — see License and Attribution for terminology/method attribution terms Repository: this repository is the canonical, timestamped source for AkbasCore, the Damped Resonance Alignment (DRA) method, the Pusula compass-construction technique, the Phase V synthetic concept-injection engine, and all associated terminology defined below.

Live test logs: ongoing test runs, per-layer activation logs, and output comparisons are actively published and timestamped in real time on two Reddit communities — r/TinyLlama_TITAN and r/TinyLlama_AGI — prior to being consolidated here. Those threads are the day-to-day development record; this repository is the stable, versioned reference.

## Table of Contents
- [What This Is](#what-this-is)
- [Why This Exists](#why-this-exists)
- [Core Idea in One Paragraph](#core-idea-in-one-paragraph)
- [Terminology (Canonical Reference)](#terminology-canonical-reference)
- [Mathematical Framework](#mathematical-framework)
- [System Architecture](#system-architecture)
- [Development History — Phase I through Phase V](#development-history-phase-i-through-phase-v)
- [Parameter Reference](#parameter-reference)
- [How to Run It](#how-to-run-it)
- [Test Methodology and Honest Limitations](#test-methodology-and-honest-limitations)
- [Relationship to Prior Work](#relationship-to-prior-work)
- [Roadmap](#roadmap)
- [License and Attribution](#license-and-attribution)
## What This Is
AkbasCore is a small, self-contained inference-time intervention layer that sits between a transformer language model's internal layers during the forward pass. It does not fine-tune, retrain, or modify any model weight. It does not alter the input prompt. It reads the hidden state at each of a model's early transformer layers, computes a small directional correction using a closed-form damped-oscillator equation, and — depending on operating mode — either logs that correction (Observer mode) or writes it into the hidden state (Steered mode).

The design goal is a form of alignment that is geometric rather than linguistic: instead of filtering tokens, blocklisting phrases, or training a classifier to gate outputs, AkbasCore defines one or more fixed directions in the model's own representation space and nudges the hidden state toward those directions by an amount that follows a critically damped decay profile across layers.

The project has two active lines. The **Phase IV line** — whose shipped artefact is **AkbasCore 1.3 DRA** — steers a 1.5B model on CPU toward an ethical Pusula built from the model's own embedding table. The **Phase V (2.1) line** is a distinct adaptation of the same damping mechanism, built for a different problem: injecting a synthetic concept vector into a 7B model on GPU through a native CUDA kernel, with the magnitude law decoupled from cosine alignment. The two lines share a philosophy, a damping envelope and much of the vocabulary; they do not share a governing equation, a model, a device, or an actuator construction. An intermediate Phase V 2.0 multi-axis branch is retained for lineage but is not the active line — see Development History.

This repository is the complete, canonical record of that method: the mathematics, the terminology, the kernel implementations, the parameter calibration, and the test logs produced while developing it — including the phases and the attempts that did not work.

## Why This Exists
Most production alignment techniques operate at one of two points: during training (RLHF, Constitutional AI, supervised fine-tuning) or at the input/output boundary (system prompts, moderation classifiers, output filters). Both are effective but come with trade-offs — training-time methods are computationally expensive and freeze behavior into the weights; boundary-time methods are comparatively easy to work around because they act on the finished token stream rather than the internal computation that produced it.

AkbasCore explores a third point: the forward pass itself. The hypothesis under test is whether a small, mathematically principled push on hidden-state geometry, applied consistently across a subset of layers, can measurably shift output character without weight modification and without the intervention needing to be large enough for standard instrumentation (cosine similarity at bfloat16 resolution) to register it as a discrete event.

This is presented as a hypothesis and an engineering experiment, not a proven alignment solution. The Honest Limitations section is a required part of this documentation, not an afterthought.

## Core Idea in One Paragraph
A unit vector — the Pusula (compass) — is constructed at model-load time by averaging the model's own token embeddings for a curated set of ethical and logical anchor words, weighted by a fixed constitution. At each of the first N transformer layers during generation, the kernel measures the cosine similarity between the current hidden state and the Pusula, computes a scalar contribution (the Katkı) using a critically damped resonance formula that starts strong and decays to a small permanent floor, and — in Steered mode — adds that scalar, scaled along the Pusula direction, to the hidden state. That is the Phase IV engine, and it remains the description of AkbasCore 1.3 DRA. As of Phase V (2.1), the same damped envelope is retained but the magnitude law is changed: at 7B scale the cosine term was found to collapse toward zero for structural reasons, so alignment is now used only as a closed-loop governor and never as the gain, the injected magnitude is set directly as a fraction of the hidden state's own norm, and the injected direction is extracted per layer from contrastive hidden states rather than from the embedding table. The decay profile, floor, overall gain and dose are user-adjustable within calibrated safe bands; the underlying formulas and construction methods are fixed architecture.

## Terminology (Canonical Reference)
All terms below are original coinages by Akbaş, introduced in this project, and are not translations of pre-existing terminology in the activation-steering or alignment literature. They are given in Turkish with fixed English glosses. Researchers extending or citing this work should use the canonical (diacritic-correct) spelling.

### Phase I–IV terms (unchanged since v1.2)
| Term | Canonical spelling | Gloss | Role |
|---|---|---|---|
| Pusula | Pusula | Compass Vector | The normalized unit direction in hidden-state space, built at runtime from the model's own embedding table, representing the target alignment direction. In Phase IV this single vector both measures alignment and is the direction written into the hidden state; Phase V splits those two roles (see below). |
| Katkı | Katkı | Contribution | The scalar value computed at each layer and (in Steered mode) added to the hidden state along a compass direction. |
| Taban | Taban | Permanent Floor | The nonzero value the resonance gain decays toward rather than decaying to zero — a deliberate departure from the classical critically damped oscillator. |
| Sönüm | Sönüm | Decay Rate | Controls how quickly the resonance peak fades across layers. |
| Zirve | Zirve | Amplitude | The initial peak strength of the resonance push at the earliest steered layer. |
| İvme | İvme | Steering Velocity | The master amplitude dial; scales all Katkı values proportionally without altering the shape of the decay curve. |
| Terazi | Terazi | Constitution Weights | The fixed weighting given to each ethical category (harm avoidance, honesty, autonomy, fairness) when constructing the ethical component of the Pusula. |
| Bileşim | Bileşim | Blend Ratio | The mixing ratio between the ethical-embedding average and the logic-anchor-embedding average used to build the final Pusula. |
| Oran | Oran | Scale Factor | A fixed multiplier applied in the Katkı formula (embedded constant, not user-adjustable). |
| Doyum | Doyum | Saturation Threshold | The upper cosine-alignment threshold above which push is reduced, preventing over-steering of already well-aligned states. |
| Karşıt | Karşıt | Counter Threshold | The lower cosine-alignment threshold below which push is amplified, for strongly misaligned states. |
| Sapma | Sapma | Uncertainty Corrector | Adjusts the effective decay rate upward when alignment is ambiguous (cosine near zero). |
| Fren | Fren | Velocity Brake | A closed-loop correction that reduces push when alignment is improving layer-over-layer and increases it when alignment is worsening. |
| Sınır | Sınır | Max Katkı Ratio | A norm-proportional ceiling that bounds the Katkı magnitude relative to the hidden-state vector's own norm. |
| Damped Resonance Alignment (DRA) | — | — | The overall method name: applying a critically damped (ζ = 1) oscillator profile, decaying to a nonzero floor, as the per-layer gain envelope of an inference-time steering kernel. |

### Phase V terms (new in v2.1 — DRA / Synthetic Injection)
These are the terms introduced by the current active line. Where a Phase V term extends, replaces or retires a Phase IV term, that relationship is stated explicitly. The Phase IV definitions above remain valid for the Phase IV engine and are not superseded retroactively.

| Term | Canonical spelling | Gloss | Role |
|---|---|---|---|
| Sensör | Sensör | Sensor Vector | The direction against which alignment cos(θ) is *measured*. In Phase IV the Pusula performed this role and the injection role at once; in Phase V the two are **split**, because measurement and injection were found to require different spaces. The Sensör operates in Mercek-masked space. |
| Aktüatör | Aktüatör | Actuator Vector | The direction actually *added* to the hidden state, in full (unmasked) space. The second half of the split above. The Pusula concept therefore does not disappear in Phase V — it divides into two independently specified objects. |
| Mercek | Mercek | Lens | A binary mask that excludes the highest-energy hidden-state dimensions from the cosine measurement only, never from the injection. Introduced because a small number of dimensions carry most of the hidden state's norm and suppress every measured alignment toward zero. First used in this phase; no Phase IV equivalent. |
| Dondurulmuş Norm | Dondurulmuş Norm | Frozen Norm (norm0) | ‖h‖ measured once, **before any injection at that layer**, and reused by every axis in that layer. Prevents axes from feeding on each other's inflation. New in Phase V. |
| Bütçe Bölüşümü | Bütçe Bölüşümü | Budget Division | Division of the total injection budget by √K across K simultaneous axes, so that total displacement remains at the İvme level for orthogonal directions. New in Phase V. |
| Zarf | Zarf | Envelope | The Phase IV damped curve **normalized to 1.00 at the first steered layer**: zarf(L) = kb(L) ⁄ kb(0). Identical Sönüm/Zirve/Taban shape as Phase IV, re-expressed as a pure shape factor once magnitude was decoupled from cos(θ). This is the direct continuation of the Phase III/IV gain envelope. |
| Kontrastif Aktüatör | Kontrastif Aktüatör | Contrastive Actuator | An Aktüatör built as the difference of mean **hidden states** — not embeddings — between a matched positive and negative sentence set, extracted separately for every layer. This is the point at which Phase V departs from the Phase IV requirement that every direction be derived from the embedding table. New in Phase V. |
| Eşleştirilmiş Negatif | Eşleştirilmiş Negatif | Paired Negative | The requirement that the negative sentence set share the positive's skeleton exactly, differing only in the target attribute. Established in this phase as the single largest lever on how discriminative an actuator turns out to be. New in Phase V. |
| Katman Başına Vektör | Katman Başına Vektör | Per-Layer Vector | One separately extracted Aktüatör per hooked layer, rather than a single vector broadcast across all layers. New in Phase V. |
| Yoğunluk | Yoğunluk | Density | Any concept counter normalized to occurrences per 100 words, adopted after raw counts were shown to shift mechanically with output length. Measurement term, new in Phase V. |
| Bozulma | Bozulma | Degradation | Symbol-salad and junk-token score. Measurement term, new in Phase V. |
| Tekrar | Tekrar | Repetition | 4-gram loop score. Catches degenerate loops that Bozulma scores as clean. Measurement term, new in Phase V. |
| Uzunluk Çöküşü | Uzunluk Çöküşü | Length Collapse | Score for output collapsing to a small fraction of baseline length. The third failure mode, invisible to the other two. Measurement term, new in Phase V. |
| Kalite | Kalite | Quality | KALİTE = max(Bozulma, Tekrar, Uzunluk Çöküşü) — the single composite gate that decides whether a run is usable. Measurement term, new in Phase V. |
| Çifte Ayrışma | Çifte Ayrışma | Double Dissociation | The control design in which two vectors are each shown to raise their own target while leaving the other's at zero, in identical context at identical dose. Evidence-design term, new in Phase V. |

**Phase IV terms whose role changed in Phase V.** The Phase IV definitions above still hold for the Phase IV engine (AkbasCore 1.3 DRA); this table records only what happened to each term inside the 2.1 line.

| Term | Phase IV role | Role in Phase V (2.1) |
|---|---|---|
| Pusula | One direction that both measured alignment and was injected | **Split** into Sensör (masked space, measurement) and Aktüatör (full space, injection). The name Pusula is retained for the Phase IV engine only. |
| İvme | Dimensionless master amplitude dial scaling Katkı proportionally | Now *directly* the fraction of ‖h‖ injected: İvme = 0.10 means a 10% injection at the first steered layer. A dimensionless dial became a physical quantity. |
| Sönüm / Zirve / Taban | Shape of the per-layer gain curve | Unchanged in value and in shape, now expressed as the normalized Zarf. |
| Fren, Doyum, Karşıt | Closed-loop velocity brake and its thresholds | **Retained in the kernel but does not engage** in the contrastive regime, because cos(θ) never reaches the Doyum threshold there. Documented as an open engineering task, not as a working feature. |
| Sınır | Norm-proportional Katkı ceiling | **Removed.** At hidden dimension 3584 it was pinned to its own cap at every layer and was the binding constraint instead of the dial. |
| Oran, Sapma | Fixed constants inside the Katkı formula | No longer present in the 2.1 magnitude law, which no longer contains the term they scaled. |

## Mathematical Framework

> **Scope note — which engine these equations describe.**
> The equations in this section are the **Phase III/IV** formulation, whose shipped artefact is **AkbasCore 1.3 DRA** (Qwen2.5-1.5B-Instruct, CPU, ethical Pusula). They are preserved here as the canonical record of that line, which is still maintained and still in use.
>
> They are **not** the governing equations of the current Phase V (2.1) engine. The 2.1 line changed the magnitude law itself: the contribution is no longer proportional to cos(θ). Anyone implementing, reimplementing or citing the current synthetic-injection engine must use the equations given under *Phase V (2.1)* below, not the Phase III/IV ones. The two engines share a damping envelope and a vocabulary; they do not share a governing equation.

### The Phase III/IV governing equation (AkbasCore 1.3 DRA)
The per-layer gain follows a critically damped oscillator (damping ratio ζ = 1):

```
P_t = cos(θ) × [ Zirve · e^(−Sönüm·t) · (1 + Sönüm·t) + Taban ]
```

Where t is the transformer layer index and cos(θ) is the real-time cosine alignment between the current hidden state and the Pusula. This decays to a nonzero permanent floor (Taban) rather than to zero: the intervention should never fully switch off at the deepest steered layer.

### Step-by-step Katkı computation (Phase III/IV)

1. **Alignment measurement.** `cos(θ) = (h · Pusula) / (‖h‖ · ‖Pusula‖)`
2. **Uncertainty-corrected decay rate.** `ω_eff = Sönüm + (1 − |cos(θ)|) × Sapma`
3. **Base layer gain.** `kb = Zirve · e^(−ω_eff·t) · (1 + ω_eff·t) + Taban`
4. **Velocity-adjusted gain (Fren).** Compares the current-layer alignment to the previous layer's:
   ```
   dr = clamp(cos(θ)_t − cos(θ)_(t−1), −0.15, +0.15)
   if dr > 0:  kv = kb × (1 − dr × Fren)
   if dr < 0:  kv = kb × (1 + |dr| × Fren)
   kv = clamp(kv, 0.05, 1.0)
   ```
5. **Norm-proportional ceiling.** `R_max = clamp(‖h‖ × Sınır, floor, ceiling)`
6. **Contribution and application.**
   ```
   son = saturation factor (Doyum/Karşıt thresholds)
   katki = clamp( İvme × cos(θ) × kv × Oran × son , −R_max, +R_max )
   h ← h + katki × Pusula
   ```

### Constructing the Pusula (Phase III/IV)

```
ethics_vec = Σ [ Terazi_c × mean(embedding(seed_words_c)) ] / Σ Terazi_c   for each category c
logic_vec  = mean(embedding(logic_anchor_words))
Pusula     = normalize( Bileşim_ethics × ethics_vec + Bileşim_logic × logic_vec )
```

Every direction is built from the target model's own real embedding table, never from externally imposed or synthetic geometry. This was a stated, absolute design constraint of the Phase IV line. Phase V found a measured reason to break it for the 7B synthetic-injection case; that finding is recorded honestly in Development History rather than applied retroactively here.

### Why a nonzero floor matters
In a classical critically damped system, the impulse response decays fully to zero. AkbasCore's Taban term means the direction is never fully released — the mathematical expression of the project's central design metaphor of an internal orientation rather than a gate that can be fully satisfied and then ignored. This remains true for every phase, including Phase V, where the envelope is retained unchanged in shape.

### Phase V (2.1) governing equation — synthetic injection engine
The 2.1 line retains the damped envelope and discards the proportionality to cos(θ). Alignment is used as a **governor**, never as a **gain**:

```
katki_k = (İvme_k / √K) × zarf(L) × norm0 × son_k
h      ← h + Σ_k katki_k · Aktüatör_k          (single accumulation)
```

| Term | Definition | Why it is there |
|---|---|---|
| `norm0` | ‖h‖, measured once **before any injection** at that layer (Dondurulmuş Norm) | Freezing it prevents axes from feeding on each other's inflation |
| `zarf(L)` | `kb(L) / kb(0)` — the Phase III/IV damped curve, normalized to 1.00 at the first steered layer | The DRA envelope is preserved intact, as a pure shape factor |
| `İvme_k` | Directly the fraction of ‖h‖ to inject | İvme = 0.10 is a 10% injection |
| `√K` | Budget division across K simultaneous axes (Bütçe Bölüşümü) | Keeps total displacement at the İvme level for orthogonal directions |
| `son_k` | Fren: cos > Doyum → (1 − cos)/0.25; cos < Karşıt → 1.6 | **The only place cos(θ) enters the formula** |

Retained constants: Sönüm 0.30, Zirve 0.70, Taban 0.20, 20 hooked layers of 28, Mercek_K 20. `Sınır` and the velocity-adjusted gain `kv` were removed after ablation showed neither was binding at this scale — capped and uncapped arms produced byte-identical output.

Because all axis contributions are computed against the same frozen `norm0` and accumulated in a single pass, multi-axis application in 2.1 is **order-independent** — in deliberate contrast to the sequential traversal used in the 2.0 branch, where each axis measured against a state the previous axis had already modified.

### Constructing the Aktüatör (Phase V)
The Aktüatör is not built from the embedding table. For each hooked layer L, over a matched positive set P and negative set N:

```
Aktüatör(L) = normalize( mean_{p∈P} h_L(p) − mean_{n∈N} h_L(n) )
```

Three construction rules, each established by measurement in this phase, are documented under Development History → Phase V: hidden states rather than embeddings; one vector per layer (Katman Başına Vektör); and a skeleton-matched Eşleştirilmiş Negatif.

## System Architecture

```
[ Tokenizer ] → [ Transformer Layer 0 ]──▶ AkbasCore hook ──▶ katkı computed and applied
                 [ Transformer Layer 1 ]──▶ AkbasCore hook ──▶ katkı computed and applied
                 [        ...          ]        ...
                 [ Transformer Layer N ]──▶ AkbasCore hook ──▶ katkı computed and applied
                 [ Layer N+1 ... final ]   (unmodified — no kernel activity)
                                  ↓
                          [ LM Head ] → [ Logits ] → [ Output tokens ]
```

The kernel registers as a forward hook on the first N transformer layers. Deeper layers are left completely untouched. Reference implementations: N = 20 of 28 layers for the Phase IV line (Qwen2.5-1.5B-Instruct, hidden dimension 1536, CPU) and N = 20 of 28 layers for the Phase V line (Qwen2.5-7B-Instruct, hidden dimension 3584, A100 GPU, bfloat16).

Two operating modes run on identical inputs for direct comparison: Observer (Vanilla — Katkı computed and logged, never written) and Steered (Katkı computed and written at every hooked layer). Running both back-to-back on the same prompt is a Dual Run.

**Phase IV kernel.** Host-side C++ (`akbas_observe`, `akbas_steer`), compiled inline via `torch.utils.cpp_extension`, driven by a Gradio interface exposing the four live motor controls. Because it is host-side, the Python harness forces every tensor the kernel touches onto CPU before the call and moves the result back afterwards, with an explicit `TORCH_CHECK` guard that raises a clear error instead of segfaulting if this is bypassed. This makes the Phase IV line correct on CPU-only environments (e.g. Termux/mobile) as well as GPU, at a measurable throughput cost on GPU.

**Phase V kernel.** A native `__global__` CUDA kernel — the one called for in the earlier roadmap, now written for this line. One block per token, shared-memory reduction, three phases: (1) compute `norm0` once in full space before any injection; (2) for each axis compute cos(θ) in Mercek-masked space and derive the scalar contribution, leaving h unmodified; (3) accumulate all axis contributions into h in a single pass. Phase (2) is what makes multi-axis application order-independent.

The method remains architecture-agnostic: any model exposing per-layer hidden states through a forward hook can be steered, provided the steering directions are reconstructed for that model.

## Development History — Phase I through Phase V
AkbasCore was developed iteratively and each phase is preserved in this repository for transparency, including the phases that did not work.

### Phase I — Linear Filter
(unchanged from prior releases) The initial formulation treated alignment as a static linear combination, P_t = (V₀ + Ω + Σφᵢ) × ε_t, a philosophical mapping exercise rather than a mechanism grounded in model internals. It was not testable against real activations and is retained here as a documented negative result.

### Phase II — Resonance Field
(unchanged from prior releases) Replaced the static formula with an undamped oscillator, P_t = cos(θ) × sin(ωt + φ), introducing a real, computable cosine-similarity term for the first time. A pure sine wave never converges to a stable operating point — mathematically unstable, but the first step from ethics-as-filter to ethics-as-field.

### Phase III — Damped Kernel (Simulational Proof of Concept)
(unchanged from prior releases) Introduced critical damping (ζ = 1) and the closed form used throughout this repository. Validated on 50 hand-constructed, manually-cosine-assigned scenarios — a deterministic simulation of the damping formula's stability, explicitly not live-model inference. This distinction is preserved deliberately as an example of correctly scoping a simulation result.

### Phase IV — Empirical Grounding and Live-Model Testing
(unchanged from prior releases) Moved from hand-assigned vectors to a Pusula constructed from a real model's own embedding table, and from a simulation harness to a compiled C++ kernel hooked into live forward passes of Qwen2.5-1.5B-Instruct. This is the phase documented by the versioned kernel (AkbasCore 1.1 → 1.2 → 1.3), the Dual Run test protocol, and the numbered test-log series. Version 1.2 added the four live-adjustable motor controls (İvme, Sönüm, Zirve, Taban) exposed through a Gradio interface. **AkbasCore 1.3 DRA is the final artefact of this phase** — ethical compass, 1.5B model, CPU — and is the system the Mathematical Framework section above describes. This line is still maintained.

### Phase V (2.0) — Multi-Axis Manifold branch (exploratory, limited yield)
An intermediate branch extended the Phase IV kernel to multiple simultaneous, Gram-Schmidt-orthogonalized axes, each with its own strength, its own layer-depth activation window and its own polarity, plus a continuous blend between the original resonance gain physics and a corrective attractor physics. It ran, it was validated across model scales up to 7B, and the three-pass measurement protocol it introduced — isolating axis 0's own effect from genuine cross-axis leakage — was sound.

It is retained here for lineage, but honestly: it did not yield a result worth building on. The cross-axis effects it was designed to exploit measured near zero and inconsistent in sign, and the multi-axis machinery did not earn its complexity. Work in this direction may resume later. It is documented rather than expanded, and its terminology has been retired from the active terminology table above.

### Phase V (2.1) — Synthetic Concept Injection Engine (current active line)
This is the current work and the line the project is now built around. It is a motor-first phase: almost everything below is about the engine — what was wrong with it, what replaced it, and what the replacement can and cannot do.

**What changed in scope.** Phase IV steers toward a *value* direction built from the model's own embedding table: an ethical compass, 1.5B model, CPU. Phase V (2.1) asks a different question on different hardware — can a **synthetic concept**, specified only as a sentence, be injected into a 7B model's forward pass in a controlled, dosable, direction-specific way? Same damping philosophy, same vocabulary, different engine.

#### The two faults that had to be fixed before anything else
Porting the Phase IV kernel to Qwen2.5-7B-Instruct (hidden dimension 3584) produced no measurable effect. Two separate causes were isolated. Both are documented because each is a reusable lesson.

**Fault 1 — the ceiling did not scale.** `Sınır` computes `clamp(‖h‖ × 0.045, 0.04, 0.20)`. At 7B the measured norms run from ‖h‖ = 13.6 at the first steered layer to 553.5 at the twentieth, so the expression evaluates between ≈0.61 and ≈24.9 — above its 0.20 cap at *every single layer*. The constant meant to scale the push with the norm was instead holding it fixed. Realized injection came out at 0.21–1.77% of ‖h‖ regardless of the İvme setting; at İvme = 1.00 the contribution was exactly +0.2000 — the ceiling, not the dial.

**Fault 2 — cosine collapse.** Removing the ceiling did not help, which was the more informative result. Measuring the sensor directly gave a mean cos(θ) of **+0.0010** — *below* the random-vector expectation of 1/√3584 = 0.0167 — and a realized injection of 0.0001% of ‖h‖. Vector provenance made no difference: a semantically meaningful direction and a synthetic Gaussian one produced statistically indistinguishable cosines.

The cause is the hidden state's own energy distribution. Measured on this model:

| Measurement | Value |
|---|---|
| Fraction of ‖h‖ carried by the top 20 of 3584 dimensions | **86.5%** |
| Largest ÷ median dimension magnitude | **428×** |
| ‖h‖ by depth | L0 = 13.6 · L8 = 488.2 · L19 = 553.5 |

A synthetic reconstruction confirmed the mechanism: with those dimensions present cos = +0.0403; with the top three excluded, cos = +0.9569. The cosine denominator is inflated by dimensions the steering vector has no reason to occupy, so alignment is mathematically suppressed toward zero. Because the Phase IV formula makes the contribution *proportional* to cos(θ), the product goes to zero.

**The engine was computing perfectly and injecting nothing.** This is the single most important engineering finding of the phase, and it generalizes: any steering rule whose magnitude is gated on cosine similarity inherits this denominator, and the problem gets worse as models get larger. The phenomenon itself is known in the literature as *massive activations*; its consequence for cosine-gated steering is what this project contributes.

**Mercek** — masking the top-energy dimensions out of the measurement only — was the first attempted fix. It produced a real 4.1× improvement (0.0010 → 0.0043) but still below chance. Correct as a diagnosis, insufficient as a treatment. It remains in the kernel because it is harmless and diagnostically useful.

#### The redesign
The fix was to stop treating cos(θ) as a gain and use it only as a governor. The resulting equation is given in full under Mathematical Framework → Phase V (2.1). In short: magnitude is set directly as a fraction of the frozen hidden-state norm, shaped by the unchanged DRA envelope and divided across axes by √K; cosine survives only inside the Fren term.

**Result: realized injection went from 0.0001% to a controlled 10.00% of ‖h‖ at the first steered layer**, decaying to 2.25% at the twentieth under the envelope. The dial is a dial again.

#### The actuator — and an honest break with a Phase IV constraint
Fixing the engine exposed that the *vector* was also wrong. Phase IV and the 2.0 branch both required every direction to come from the model's own embedding table, without exception. **On 7B, embedding-derived actuators do not transport concepts.** Measured against a contrastive hidden-state actuator for the same concept, the cosine between the two constructions is **−0.0017** — they are orthogonal. These are not two estimates of one direction; they are unrelated directions, and only one of them is where the residual stream actually represents the concept.

This is a direct refutation of a design commitment stated in earlier releases of this README. It is recorded as such rather than quietly dropped. The 2.1 line builds actuators from per-layer contrastive hidden states, under three rules, each supported by a measurement:

1. **Hidden states, not the embedding table.** Target-transfer counter at the same dose: embedding actuator 0.0, contrastive actuator 6.2.
2. **One vector per layer** (Katman Başına Vektör). cos(L0, L6) = +0.1759; cos(L0, L19) = +0.0516. A single vector broadcast across twenty layers is, for most of them, close to a random direction.
3. **Eşleştirilmiş Negatif.** The negative must share the positive's skeleton exactly. Holding the positive fixed and varying only the negative moved the net target score from −0.8 (generic negative) to **+5.8** (skeleton-matched). With a generic negative the difference vector retains a large shared component — cosine +0.4958 to an entirely unrelated concept's vector — which dominates and washes out the discriminative part.

#### What the engine demonstrably does
- **Dosable injection.** İvme is realized to two decimal places at the first steered layer.
- **Dose-response.** Target density rises monotonically with dose across the usable band, while a random-vector control sits *below* baseline at every dose — a directional effect, not an energy effect.
- **Direction specificity, by Çifte Ayrışma.** In identical context at identical dose: actuator A gives target A 8.2 and target B 0.0; actuator B gives target A 0.0 and target B 7.2, while also pushing its anti-target *down* from 2.8 to 1.5. Each vector raises its own target and leaves the other at exactly zero.
- **Byte-level reproducibility.** Three independent OS processes with fresh CUDA contexts and independent model loads produced character-for-character identical output and identical per-prompt quality to three decimals (across-session σ = 0.000).
- **A mapped operating band.** İvme 0.06–0.10 stays inside the quality gate; the safe production value is locked at **İvme = 0.10**.

#### What the engine demonstrably does not do — measured boundaries
Reported at the same level of detail as the successes, because a control mechanism with an unknown envelope is not usable by anyone else.

- **Context dependence.** Same actuator, same dose, two framings: transfer score 8.2 in a supportive context versus 1.5 in a neutral one. Steering does not *write* a concept; it biases a selection among continuations the context has already made available. A steering vector's effect size is therefore **not a property of the vector**.
- **The Fren does not engage in this regime.** The brake fires only above cos = 0.75, which contrastive actuators never reach. Measured brake-on versus brake-off difference: exactly 0.000, in four independent runs. The one run in which the brake *did* win decisively (degradation 0.069 with brake versus 0.846 without, at İvme 0.25) used the same vector as both Sensör and Aktüatör — an echo configuration where cos reached 0.82. That result is real but **conditional**, and is reported as such.
- **Geometric similarity does not predict functional similarity.** Eight leave-one-out variants of one actuator had cosine **0.9950 ± 0.0011** to the full vector — by any vector-space criterion, the same direction — yet their behavioural effect ranged from 7.7 to 1.8, a factor of four. Validating a steering vector by cosine agreement is not sufficient.
- **Composition of separately-extracted vectors failed under every method tried.** Sequential application produced compound over-injection and degenerate repetition loops; Bütçe Bölüşümü with Dondurulmuş Norm prevented the collapse but lost *both* concepts; in-kernel orthogonalization was a no-op (cos between original and orthogonalized vector +0.9927, axes already near-orthogonal at +0.0465, output unchanged). What worked was extracting a **single** actuator from one jointly-specified positive/negative sentence pair. Composition appears to be a property of extraction, not of application.
- **A safety/efficacy tension, unresolved.** At the safe dose the effect over baseline is small once length is corrected for; the clearly robust effect appears at a dose that fails the quality gate. Widening that band is the headline engineering task for the next version.

#### Measurement instrumentation (reusable, independent of this engine)
Three times in this phase a metric reported a conclusion that reading the generated text refuted. Each taught a distinct degradation mode, and all three are now required:

| Mode | What it looks like | What misses it |
|---|---|---|
| Bozulma | Symbol salad, junk tokens, script drift | — |
| Tekrar | *"He is the man who is the man"* × 20 | Bozulma scored this **0.006** — the cleanest-looking arm in the run |
| Uzunluk Çöküşü | Sixty words become five | Both of the above return ≈0 on a five-word string |

KALİTE = max(Bozulma, Tekrar, Uzunluk Çöküşü); sound < 0.15, suspect < 0.40, broken > 0.40. All concept counters are Yoğunluk (per 100 words), because raw counts shift mechanically with length.

A fourth instrumentation finding is worth stating separately: **a keyword lexicon can manufacture an effect.** Four generic words in a target list caused a **random** vector to score 0.50 on that target in a scene context, indistinguishable from the real vector's 0.69. Removing those four words dropped both to 0.00. The random-vector control arm is what exposed this, which is an argument for always running one.

> **Operating rule, learned three times.** Whatever the counter says, read the text. A counter read 1.8 for a concept while the text described something else entirely. A degradation score read 0.006 on text repeating itself twenty times. A density delta read −6.8 on text that had in fact written the target scene.

#### Status
23 controlled runs, every one published as a matched pair — the exact executed script and its complete unedited output log. The engine runs, its effect is measurable and direction-specific, and its current envelope is mapped. The full write-up, including the mathematics, the literature positioning and the complete limitations, is published as a pre-print in this repository.

This is a first-generation engine. The control surface it provides — how much, decaying how fast, to what floor, over which depth, with what closed-loop response — is larger than what has been used with it so far. This is the current and active phase of the project, and the next steps are specified in Roadmap.

## Parameter Reference
### Phase IV controls (AkbasCore 1.3 DRA — adjustable per run, within calibrated safe bands)
| Parameter | Meaning | Default | Safe band | Since |
|---|---|---|---|---|
| İvme | Steering velocity — master amplitude | 0.50 | 0.20 – 0.80 | Phase IV |
| Sönüm | Decay rate — how fast the push fades across layers | 0.30 | 0.10 – 0.60 | Phase IV |
| Zirve | Amplitude — initial peak push at early layers | 0.70 | 0.30 – 1.00 | Phase IV |
| Taban | Permanent floor — residual pressure at deep layers | 0.20 | 0.05 – 0.40 | Phase IV |
| Sıcaklık (standard sampling parameter, not an original coinage) | Output sampling randomness; 0 = deterministic/greedy | 0.65 | 0.00 – 1.50 | Phase IV |

### Phase IV embedded constants (fixed in the compiled kernel)
| Parameter | Meaning | Value |
|---|---|---|
| Oran | Base scale factor in the Katkı formula | 0.32 |
| Doyum | Saturation upper threshold | 0.75 |
| Karşıt | Counter (misalignment) threshold | −0.40 |
| Sapma | Uncertainty correction weight | 0.20 |
| Fren | Velocity brake coefficient | 0.30 |
| Sınır | Norm-proportional Katkı ceiling ratio | 0.045 (clamped to [0.04, 0.20]) |
| Bileşim | Ethics/logic blend ratio in Pusula construction | 0.40 / 0.60 |

### Phase V (2.1) parameters — synthetic injection engine
| Parameter | Meaning | Value | Notes |
|---|---|---|---|
| İvme | Fraction of ‖h‖ injected at the first steered layer | **0.10** | Locked production value. Usable band 0.06 – 0.10; the quality gate is crossed between 0.10 and 0.12 |
| Sönüm / Zirve / Taban | Envelope shape, as Phase IV | 0.30 / 0.70 / 0.20 | Unchanged; expressed as normalized Zarf(L) = kb(L)/kb(0) |
| Mercek_K | Top-energy dimensions masked from the cosine measurement | 20 | Measurement only; injection is always full-space |
| Hooked layers | Layers on which the kernel is active | L0 – L19 of 28 | L20 – L27 untested at this scale |
| K | Simultaneous axes | 1 – 4 | Budget divided by √K (Bütçe Bölüşümü) |
| Doyum / Karşıt | Fren thresholds, retained | 0.75 / −0.40 | Does not engage in the contrastive regime — see Phase V |
| Sınır | — | **removed** | Pinned to its own cap at every layer at 3584 dimensions |
| kv (velocity-adjusted gain) | — | **removed** | Not binding at this scale; capped and uncapped arms byte-identical |
| Sampling | Decoding configuration | greedy | `do_sample=False`, `repetition_penalty=1.15`, `max_new_tokens=260`, seed 42 |

## How to Run It
**Phase IV line (AkbasCore 1.3 DRA).** The kernel is distributed as a single self-contained script (see repository file list) that:

1. Installs required dependencies (ninja, gradio, bitsandbytes).
2. Compiles the C++ kernel inline via torch.utils.cpp_extension.
3. Loads the target model and constructs the Pusula from its embedding table.
4. Launches a Gradio interface exposing the motor sliders, the sampling-temperature control, and a Dual Run button.

Requirements: Python 3.7+, PyTorch, a Hugging Face transformers-compatible causal LM. No external dependencies beyond the standard scientific Python stack; runs on CPU (with correspondingly longer generation times) or GPU.

```bash
# In a fresh environment (e.g. Google Colab):
# 1. Paste the full kernel script into a single cell.
# 2. Run the cell — it will install dependencies, compile the kernel,
#    load the model, and print a Gradio share link.
# 3. Open the link, set the motor sliders, enter a prompt, and press
#    "Dual Run" to get paired Vanilla / Steered outputs plus full
#    per-layer activation logs for both passes.
```

**Phase V line (2.1).** Each published experiment is a single self-contained cell that compiles the CUDA kernel inline, loads Qwen2.5-7B-Instruct in bfloat16, extracts the per-layer contrastive Aktüatör from the positive/negative sentence sets defined at the top of the script, and runs the matched baseline/steered/random-control arms under greedy decoding. Every run in the record is published as a matched pair: the exact executed script and its complete unedited output log.

## Test Methodology and Honest Limitations
This section is a required part of the documentation, not a disclaimer appended for legal reasons.

**What the Dual Run protocol demonstrates.** Running Observer and Steered mode on an identical prompt, with identical sampling parameters, isolates the effect of the hidden-state intervention from all other sources of output variation. Differences between the two outputs are attributable to the Katkı being written into the hidden state — provided sampling temperature is controlled for.

**What "sub-threshold" means and does not mean.** (unchanged from Phase IV) At default settings, individual per-layer Katkı values fall below the bfloat16 numeric resolution floor. This is a real, verifiable property of floating-point arithmetic — it is not evidence, by itself, that the intervention meaningfully changed model behavior. Behavioral change must be established independently by comparing actual outputs.

**What remains unproven — Phase IV items (unchanged):**

- The Pusula-to-hidden-state mapping has not been pre-trained or validated against a labeled ethical-contrast dataset; it is a fixed geometric construction, not a learned classifier.
- No formal ablation study across a standardized benchmark has been run to quantify capability trade-offs.
- Sample sizes in the published test-log series are small and single-operator; they demonstrate the mechanism operates as designed, not a statistically powered behavioral study.
- Phase III validation numbers are simulation-only and should never be cited as live-model results.

**What remains unproven — Phase V (2.1) items:**

- **One model, one family, one size.** Every 2.1 measurement is Qwen2.5-7B-Instruct, bfloat16, layers 0–19 of 28. The massive-activation mechanism is documented across architectures, so the failure mode is expected to generalize; the specific numbers are not claimed to.
- **Automated lexical counters, no human evaluation.** Concept presence is measured by keyword density, which measures lexical presence, not coherent representation.
- **Single-operator, single-seed.** Reproducibility is established across processes at fixed seed, not across seeds — greedy decoding makes the former exact and leaves the latter open.
- **The brake is untested in the regime it was designed for.** Its one decisive result is conditional on an echo configuration and should not be cited as a general property.
- **The safe dose and the clearly effective dose do not currently overlap.** This is stated as an unresolved tension, not as a solved safety property.
- **Cosine agreement between steering vectors is not sufficient validation**, per the leave-one-out result — including for vectors produced by this project's own methods.
- **Composition beyond a single jointly-extracted actuator is unsolved.**
- **Layers 20–27 are untouched** at this scale, and no claim is made about them.

**Where the raw logs live.** (unchanged) Every individual test run — full per-layer activation logs, vanilla/steered output pairs, and settings — is posted publicly and chronologically to r/TinyLlama_TITAN and r/TinyLlama_AGI as it happens, and consolidated in this repository. This README summarizes and organizes that record but does not replace it.

**Why this section exists.** (unchanged) A system that cannot state what it has not yet shown is not trustworthy on the things it has shown.

## Relationship to Prior Work
- **Representation Engineering (Zou et al., 2023)** — established that linear directions in a model's hidden-state space correlate with, and can be used to influence, high-level behaviors. AkbasCore's Pusula construction is a runtime, embedding-table-derived variant of this idea; the Phase V contrastive actuator is a difference-of-means construction in the same technique family, and is documented as such rather than as an original contribution.
- **Activation Addition (Turner et al., 2023)** — demonstrated that adding a fixed vector to a single layer's residual stream produces consistent behavioral shifts. AkbasCore differs by applying a time-varying, closed-loop gain across many layers rather than one fixed-magnitude addition at one layer.
- **Constitutional AI (Anthropic, 2022)** — applies stated principles to shape behavior through training-time critique and revision. AkbasCore explores an analogous constitutional structure (the Terazi weights) expressed as a purely inference-time geometric direction.
- **Massive activations (Sun et al., 2024)** — documented that a small number of input-agnostic dimensions carry activation magnitudes orders of magnitude larger than the rest. This project rediscovered the phenomenon independently while diagnosing a dead engine. The contribution claimed here is not the observation but its **consequence**: massive activations are a structural reason why any steering rule that gates magnitude on cosine similarity will fail silently, and will fail worse as models scale. The practical rule derived from it — *use alignment as a governor, never as a gain* — is the basis of the Phase V (2.1) governing equation.
- **Steering-vector reliability (Tan et al., 2024)** — found steerability highly variable across inputs, brittle to prompt changes, and sometimes anti-steerable. This project's context-dependence result is consistent with that and offers a partial mechanism for it. The leave-one-out result sharpens a related observation in the opposite direction: where follow-up work found same-concept vectors with *low* mutual cosine behaving alike, this work documents vectors at cosine **0.995** whose effects differ **4×** — the more dangerous case, because low cosine at least warns you.
- **SAE feature steering (Templeton et al., 2024)** — clamping learned monosemantic features achieved a degree of concept dominance that difference-of-means directions did not reach here, and this is stated plainly rather than elided. What the 2.1 line offers alongside it is different in kind: no dictionary-learning stage, and a **dose-control mechanism** — damped envelope plus closed-loop brake — that is orthogonal to how the direction was found. Applying the envelope to SAE-derived directions is untested and is the most promising combination this work suggests.
- Critical damping is a standard result in classical control theory; its application here — as the gain envelope of a transformer steering kernel, decaying to a nonzero floor — remains the specific novel contribution documented under the name Damped Resonance Alignment. The Phase V magnitude-decoupled formulation, in which alignment acts solely as a governor and the injected magnitude is set directly as a fraction of the frozen hidden-state norm, is the specific novel contribution of this release.

A note on naming: earlier drafts of this project used the working name "TITAN." That name has been retired in this repository to avoid any confusion with the unrelated, independently published memory-architecture paper "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2024). AkbasCore is the sole name used for this system going forward. Separately, an early Phase V experimental branch used the informal working name "Sector"; that name is retired for the same reason it is documented — to prevent an abandoned, unreleased approach from being confused with the shipped architecture.

## Roadmap
### Phase V (2.1) — active engineering line
- **Make the Fren engage in the contrastive regime.** It currently fires only above cos = 0.75, which contrastive actuators never reach. Either derive the threshold from the observed cosine distribution per run, or re-anchor the brake on accumulated katki/norm0 rather than on alignment. This converts the project's one conditional result into a general one and is the highest-value single change available.
- **Widen the band between the safe dose and the effective dose.** Candidate levers: better-conditioned actuators, depth-selective dosing instead of a uniform İvme, and the working brake above.
- **Apply the DRA envelope to directions found by other methods**, including SAE-derived features — testing whether dose control and direction quality are genuinely separable contributions.
- **Establish the scale story quantitatively.** The massive-activation mechanism predicts that cosine-gated steering degrades with model size while magnitude-decoupled steering does not. Running both formulations across 1.5B, 7B and larger under one protocol would turn a mechanism into a law.
- **Multi-concept composition at the extraction stage** — one jointly-extracted actuator worked where four application-side methods failed; whether this generalizes is open.
- **Layers 20–27**, never touched at this scale.
- **Human evaluation**, to replace automated keyword density as the primary concept measure.

### Carried forward from earlier phases
- Empirical validation of the Pusula mapping against a labeled ethical-contrast dataset (Phase IV line).
- Capability-preservation benchmarking (MMLU, HumanEval, or comparable) across a range of İvme/Zirve settings.
- Adversarial and jailbreak-prompt evaluation of the cosine-alignment and Fren/Karşıt response.
- Multi-model validation beyond the current Qwen2.5-1.5B / Qwen2.5-7B / TinyLlama-1.1B reference implementations.
- Possible resumption of the Phase V 2.0 multi-axis line, if a use case justifies its complexity.
- Formal write-up and submission for peer review. *(A pre-print for the 2.1 line is published in this repository; peer review has not yet been sought.)*

## License and Attribution
### License Scope — What MIT Covers and What It Does Not
This project uses two separate licensing scopes, and they are not interchangeable:

✅ **Covered by the MIT License** — the software artifacts only:

- The C++ and CUDA kernel source (akbas_observe, akbas_steer, the Phase V `__global__` injection kernel, and related compiled extension code)
- The Python harness (model loading, hook registration, Dual Run orchestration, actuator extraction)
- The Gradio interface code
- Any other executable code file in this repository

You may use, copy, modify, and redistribute these code files under standard MIT terms, including commercially, provided the copyright notice is retained.

❌ **NOT covered by the MIT License** — the intellectual content of the method itself:

- The Damped Resonance Alignment (DRA) method and its governing equations
- The Pusula runtime-construction technique
- The Phase V magnitude-decoupled formulation — `katki_k = (İvme_k / √K) × zarf(L) × norm0 × son_k` — in which cosine alignment acts solely as a governor and never as a gain, together with the Sensör/Aktüatör split, Mercek, Dondurulmuş Norm and Bütçe Bölüşümü
- The full terminology set defined in this document:
  - **Phase I–IV:** Pusula, Katkı, Taban, Sönüm, Zirve, İvme, Terazi, Bileşim, Oran, Doyum, Karşıt, Sapma, Fren, Sınır
  - **Phase V (2.1):** Sensör, Aktüatör, Mercek, Dondurulmuş Norm, Bütçe Bölüşümü, Zarf, Kontrastif Aktüatör, Eşleştirilmiş Negatif, Katman Başına Vektör, Yoğunluk, Bozulma, Tekrar, Uzunluk Çöküşü, Kalite, Çifte Ayrışma
- The parameter calibration values and safe-band ranges
- The critically-damped-with-nonzero-floor formulation described in Mathematical Framework

These are original scientific and mathematical contributions, not software, and MIT licensing (which governs source code copying) does not apply to them. Reimplementing this method in a different programming language, under a different name, or as part of a different codebase does not remove the attribution requirement. Note: standard, pre-existing techniques used in this project — word-set mean pooling, difference-of-means direction construction, token sampling, and sampling temperature — are explicitly excluded from this attribution requirement, as they are not original contributions of this project.

In short: you are free to fork, run, and build on the code. You are not free to present the DRA method, the Pusula construction technique, the Phase V magnitude-decoupled formulation, or the terminology above as your own original contribution without attribution.

Method, terminology, and mathematical framework: the Damped Resonance Alignment method (Phase III/IV) and the Phase V synthetic concept-injection engine, the Pusula runtime-construction technique, and the complete original terminology defined in this document are original contributions first published in this repository by Akbaş, June 2026 (DRA) and September 2026 (Phase V). These are documented here, with full mathematical specification, a public timestamped commit history, and a parallel public test-log record on r/TinyLlama_TITAN and r/TinyLlama_AGI, as the canonical reference for this method and terminology.

Suggested citation:

```
Akbaş. AkbasCore: Damped Resonance Alignment — A Sub-Threshold
Inference-Layer Steering Framework for Transformer Models.
GitHub repository, v2.1, September 2026.
https://github.com/ceceli33/titan-cognitive-core

Akbaş. Damped Resonance Alignment: Closed-Loop Layerwise Activation
Steering and the Boundary Conditions of Concept Injection.
Pre-print, September 2026.
```

Base models used in development remain under their own original licenses and are unmodified by this work:

- TinyLlama-1.1B-Chat-v1.0 — Apache 2.0
- Qwen2.5-1.5B-Instruct — Apache 2.0 (Alibaba Cloud, Qwen Team) — Phase IV reference implementation, AkbasCore 1.3 DRA
- Qwen2.5-7B-Instruct — Apache 2.0 (Alibaba Cloud, Qwen Team) — Phase V reference implementation

AkbasCore does not alter the weights of any base model. All steering occurs at inference time via forward hooks; the underlying models remain fully and separately licensed under their respective terms.
