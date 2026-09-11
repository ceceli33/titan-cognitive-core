AkbasCore is a sub-threshold, inference-layer steering framework for transformer models. It employs Damped Resonance Alignment (DRA) to mathematically adjust a model's hidden states during the forward pass—enabling precise, modular alignment without fine-tuning, weight modification, or prompt engineering. This repository is the canonical record for the DRA methodology, the Pusula compass-construction technique, the native multi-axis Manifold extension, and all associated original terminology.

# AkbasCore — Damped Resonance Alignment (DRA)
*A sub-threshold, inference-layer activation steering framework for transformer language models.*

**Author:** Akbaş
**Status:** Independent research pre-print — not yet peer-reviewed
**Version:** 2.0 — DRA/MANIFOLD (September 2026) · baseline 1.2 (June 2026) remains documented below for lineage
**License:** MIT (kernel code) — see [License and Attribution](#license-and-attribution) for terminology/method attribution terms
**Repository:** this repository is the canonical, timestamped source for AkbasCore, the Damped Resonance Alignment (DRA) method, the Pusula compass-construction technique, the native multi-axis Manifold extension, and all associated terminology defined below.

**Live test logs:** ongoing test runs, per-layer activation logs, and output comparisons are actively published and timestamped in real time on two Reddit communities — r/TinyLlama_TITAN and r/TinyLlama_AGI — prior to being consolidated here. Those threads are the day-to-day development record; this repository is the stable, versioned reference.

## Table of Contents
- [What This Is](#what-this-is)
- [Why This Exists](#why-this-exists)
- [Core Idea in One Paragraph](#core-idea-in-one-paragraph)
- [Terminology (Canonical Reference)](#terminology-canonical-reference)
- [Mathematical Framework](#mathematical-framework)
- [System Architecture](#system-architecture)
- [Development History — Phase I through Phase V](#development-history--phase-i-through-phase-v)
- [Parameter Reference](#parameter-reference)
- [How to Run It](#how-to-run-it)
- [Test Methodology and Honest Limitations](#test-methodology-and-honest-limitations)
- [Relationship to Prior Work](#relationship-to-prior-work)
- [Roadmap](#roadmap)
- [License and Attribution](#license-and-attribution)

## What This Is

AkbasCore is a small, self-contained inference-time intervention layer that sits between a transformer language model's internal layers during the forward pass. It does not fine-tune, retrain, or modify any model weight. It does not alter the input prompt. It reads the hidden state at each of a model's early transformer layers, computes a small directional correction using a closed-form damped-oscillator equation, and — depending on operating mode — either logs that correction (Observer mode) or writes it into the hidden state (Steered mode).

The design goal is a form of alignment that is geometric rather than linguistic: instead of filtering tokens, blocklisting phrases, or training a classifier to gate outputs, AkbasCore defines one or more fixed directions in the model's own representation space (built from the model's own embedding table) and nudges the hidden state toward those directions by an amount that follows a critically damped decay profile across layers.

As of Phase V (version 2.0), the framework supports multiple simultaneous, mutually orthogonal steering directions — not just the single Pusula of the 1.2 baseline — while preserving the original DRA gain envelope and the requirement that every direction be derived from real model embeddings, never from synthetic or externally imposed geometry.

This repository is the complete, canonical record of that method: the mathematics, the terminology, the C++ kernel implementation, the parameter calibration, and the test logs produced while developing it.

## Why This Exists

Most production alignment techniques operate at one of two points: during training (RLHF, Constitutional AI, supervised fine-tuning) or at the input/output boundary (system prompts, moderation classifiers, output filters). Both are effective but come with trade-offs — training-time methods are computationally expensive and freeze behavior into the weights; boundary-time methods are comparatively easy to work around because they act on the finished token stream rather than the internal computation that produced it.

AkbasCore explores a third point: the forward pass itself. The hypothesis under test is whether a small, mathematically principled push on hidden-state geometry, applied consistently across a subset of layers, can measurably shift output character without weight modification and without the intervention needing to be large enough for standard instrumentation (cosine similarity at bfloat16 resolution) to register it as a discrete event.

This is presented as a hypothesis and an engineering experiment, not a proven alignment solution. The Honest Limitations section is a required part of this documentation, not an afterthought.

## Core Idea in One Paragraph

A unit vector — the Pusula (compass) — is constructed at model-load time by averaging the model's own token embeddings for a curated set of ethical and logical anchor words, weighted by a fixed constitution. At each of the first N transformer layers during generation, the kernel measures the cosine similarity between the current hidden state and the Pusula, computes a scalar contribution (the Katkı) using a critically damped resonance formula that starts strong and decays to a small permanent floor, and — in Steered mode — adds that scalar, scaled along the Pusula direction, to the hidden state. As of Phase V, additional directions (Eksen) can be constructed the same way — always from real embeddings — and are made mutually orthogonal to the Pusula and to each other before being applied in the same per-layer loop, each with its own strength and its own layer-depth activation window. The decay profile, floor, overall gain, axis strengths, and activation windows are user-adjustable within calibrated safe bands; the underlying formulas and compass-construction method are fixed architecture.

## Terminology (Canonical Reference)

All terms below are original coinages by Akbaş, introduced in this project, and are not translations of pre-existing terminology in the activation-steering or alignment literature. They are given in Turkish with fixed English glosses. Researchers extending or citing this work should use the canonical (diacritic-correct) spelling.

### Phase I–IV terms (unchanged since v1.2)

| Term | Canonical spelling | Gloss | Role |
|---|---|---|---|
| Pusula | Pusula | Compass Vector | The normalized unit direction in hidden-state space, built at runtime from the model's own embedding table, representing the target alignment direction. As of Phase V this is also referred to as **axis 0**, the fixed safety/logic reference that every additional axis is orthogonalized against. |
| Katkı | Katkı | Contribution | The scalar value computed at each layer and (in Steered mode) added to the hidden state along a compass direction. |
| Taban | Taban | Permanent Floor | The nonzero value the resonance gain decays toward rather than decaying to zero — a deliberate departure from the classical critically damped oscillator. |
| Sönüm | Sönüm | Decay Rate | Controls how quickly the resonance peak fades across layers. |
| Zirve | Zirve | Amplitude | The initial peak strength of the resonance push at the earliest steered layer. |
| İvme | İvme | Steering Velocity | The master amplitude dial; scales all Katkı values proportionally without altering the shape of the decay curve. As of Phase V, İvme specifically scales axis 0 (Pusula); additional axes have their own independent strength dial (see Eksen Gücü below). |
| Terazi | Terazi | Constitution Weights | The fixed weighting given to each ethical category (harm avoidance, honesty, autonomy, fairness) when constructing the ethical component of the Pusula. |
| Bileşim | Bileşim | Blend Ratio | The mixing ratio between the ethical-embedding average and the logic-anchor-embedding average used to build the final Pusula. |
| Oran | Oran | Scale Factor | A fixed multiplier applied in the Katkı formula (embedded constant, not user-adjustable). |
| Doyum | Doyum | Saturation Threshold | The upper cosine-alignment threshold above which push is reduced, preventing over-steering of already well-aligned states. |
| Karşıt | Karşıt | Counter Threshold | The lower cosine-alignment threshold below which push is amplified, for strongly misaligned states. |
| Sapma | Sapma | Uncertainty Corrector | Adjusts the effective decay rate upward when alignment is ambiguous (cosine near zero). |
| Fren | Fren | Velocity Brake | A closed-loop correction that reduces push when alignment is improving layer-over-layer and increases it when alignment is worsening. As of Phase V, each axis maintains its own independent Fren state, so one axis's alignment history never contaminates another's. |
| Sınır | Sınır | Max Katkı Ratio | A norm-proportional ceiling that bounds the Katkı magnitude relative to the hidden-state vector's own norm. |
| Damped Resonance Alignment (DRA) | — | — | The overall method name: applying a critically damped (ζ = 1) oscillator profile, decaying to a nonzero floor, as the per-layer gain envelope of an inference-time steering kernel. |

### Phase V terms (new in v2.0 — DRA/MANIFOLD)

| Term | Canonical spelling | Gloss | Role |
|---|---|---|---|
| Eksen | Eksen | Axis | An independent steering direction, built from real embeddings, carrying its own strength, its own layer-depth activation window, and its own polarity. Axis 0 is always the Pusula; up to three additional axes are supported per run. |
| Kutup | Kutup | Polarity | Whether an axis is Çekici (attractor) or İtici (repulsor). |
| Çekici | Çekici | Attractor | A polarity/physics mode in which the hidden state is pulled toward an axis's direction regardless of the state's current alignment sign — magnitude scales with Açısal Hata, direction never flips. |
| İtici | İtici | Repulsor | A polarity in which the hidden state is pushed away from a concept. Implemented as Çekici attraction to the negated axis vector — repulsion from *v* is mathematically identical to attraction toward *−v*; no separate physics was added to the kernel for this. |
| Rezonans | Rezonans | Resonance (gain physics) | The original Phase III/IV DRA gain physics, generalized: the sign of the contribution follows the sign of cos(θ), amplifying whichever alignment the hidden state already has rather than correcting it. |
| Yönlendirme Fiziği | Yönlendirme Fiziği | Steering Physics | A continuous dial (0→1) blending Rezonans and Çekici gain physics for every axis in a run. At 0, all axes behave exactly as the Phase III/IV amplifier; at 1, all axes behave as corrective attractors. |
| Açısal Hata | Açısal Hata | Angular Error | (1 − cos θ) ⁄ 2 ∈ [0, 1] — the normalized angular distance between the current hidden state and a target axis. The Çekici contribution magnitude is proportional to this term and is zero only when the state is already perfectly aligned. |
| Merkez | Merkez | Center (layer index) | The transformer layer index at which a non-Pusula axis's Katman-Derinlik Eğrisi begins ramping up from 0. |
| Genişlik | Genişlik | Width (layers) | The number of layers over which a non-Pusula axis's activation ramps from fully inactive to fully active. |
| Katman-Derinlik Eğrisi | Katman-Derinlik Eğrisi | Layer-Depth Curve | The piecewise-linear ramp function, parameterized by Merkez and Genişlik, that lets an axis stay silent through early (syntax-dominant) layers and activate through deep (intent/tone-dominant) layers. Axis 0 (Pusula) is exempt and remains at full strength at every hooked layer, matching Phase III/IV behavior exactly. |
| Dikleştirme | Dikleştirme | Orthogonalization | The native, in-kernel Gram-Schmidt process that makes every additional axis perpendicular to the Pusula and to every other active axis before any Katkı is computed. Null-space separation from the Pusula direction is a direct mathematical consequence of this step, not a separate projection stage. |

> **Note on scope of this table.** A small number of Phase V parameter and method *names* (word-set-mean axis construction, difference-of-means axis construction, token-sampling axis construction, and exposing sampling temperature as a run-time control) are Turkish/English labels for **pre-existing, standard techniques** — difference-of-means in particular is the same method cited under Zou et al. (2023) in [Relationship to Prior Work](#relationship-to-prior-work). Per this table's own inclusion criterion, they are documented in [Parameter Reference](#parameter-reference) but are **not** listed above and are **not** claimed as original coinages.

## Mathematical Framework

### The governing equation (unchanged since Phase III/IV)

The per-layer gain for any axis still follows a critically damped oscillator (damping ratio ζ = 1):

```
P_t = cos(θ) × [ Zirve · e^(−Sönüm·t) · (1 + Sönüm·t) + Taban ]
```

Where *t* is the transformer layer index and cos(θ) is the real-time cosine alignment between the current hidden state and the axis in question. This still decays to a nonzero permanent floor (Taban) rather than to zero, for the same reason documented in Phase III/IV: the intervention should never fully switch off at the deepest steered layer.

### Step-by-step Katkı computation, per axis (Phase III/IV steps, unchanged)

1. **Alignment measurement.** `cos(θ) = (h · axis) / (‖h‖ · ‖axis‖)`
2. **Uncertainty-corrected decay rate.** `ω_eff = Sönüm + (1 − |cos(θ)|) × Sapma`
3. **Base layer gain.** `kb = Zirve · e^(−ω_eff·t) · (1 + ω_eff·t) + Taban`
4. **Velocity-adjusted gain (Fren).** Compares this axis's current-layer alignment to its own previous-layer alignment (each axis keeps an independent history as of Phase V):
   ```
   dr = clamp(cos(θ)_t − cos(θ)_(t−1), −0.15, +0.15)
   if dr > 0:  kv = kb × (1 − dr × Fren)
   if dr < 0:  kv = kb × (1 + |dr| × Fren)
   kv = clamp(kv, 0.05, 1.0)
   ```
5. **Norm-proportional ceiling.** `R_max = clamp(‖h‖ × Sınır, floor, ceiling)`
6. **Gain-physics blend (new in Phase V).** Two candidate contributions are computed and blended by the Yönlendirme Fiziği dial *m* ∈ [0, 1]:
   ```
   son = saturation factor (Doyum/Karşıt thresholds, as in Phase III/IV)
   katki_rezonans  = İvme_or_Eksen_Gücü × cos(θ) × kv × Oran × son
   açısal_hata     = (1 − cos(θ)) / 2
   katki_çekici    = İvme_or_Eksen_Gücü × kv × Oran × açısal_hata × 2
   k_t = clamp( (1 − m) × katki_rezonans + m × katki_çekici, −R_max, +R_max )
   ```
   At *m* = 0 this reduces exactly to the Phase III/IV formula. At *m* = 1, sign no longer follows cos(θ): the contribution is always directed toward the axis (or away from it, if Kutup = İtici, via a negated axis vector), with magnitude proportional to how far the state currently is from that direction.
7. **Layer-depth gating (new in Phase V).** For axis 0 (Pusula), `curve = 1` at every layer, exactly matching Phase III/IV. For every additional axis *k*:
   ```
   curve_k(L) = clamp( (L − (Merkez_k − Genişlik_k / 2)) / Genişlik_k , 0, 1 )
   k_t ← k_t × curve_k(L)
   ```
8. **Sequential application across axes.** For a run with K axes (axis 0 = Pusula plus up to three additional axes), the contributions are applied one axis at a time to the *same* hidden-state vector, in order:
   ```
   for k in 0..K−1:
       h ← h + k_t(axis_k) × axis_k
   ```
   Each axis therefore measures its cosine alignment against the state as already modified by every prior axis in the same layer — a sequential traversal of the manifold rather than K independent, non-interacting pushes.

### Constructing the Pusula and additional axes

The Pusula construction is unchanged from Phase III/IV:

```
ethics_vec = Σ [ Terazi_c × mean(embedding(seed_words_c)) ] / Σ Terazi_c   for each category c
logic_vec  = mean(embedding(logic_anchor_words))
Pusula     = normalize( Bileşim_ethics × ethics_vec + Bileşim_logic × logic_vec )
```

Additional axes (Phase V) are built by one of three methods — all, without exception, from the target model's own real embedding table, never from externally imposed or synthetic geometry:

- **Word-set mean:** the normalized mean embedding of a single concept word list (e.g. *contradiction, chaos, nonsense, vague, hallucination, incoherent, illogical, absurd*).
- **Difference-of-means:** the normalized difference between two concept word-list mean embeddings (the same family of method as Zou et al., 2023 — see [Relationship to Prior Work](#relationship-to-prior-work)).
- **Seeded token sampling:** the normalized mean embedding of a seeded sample of real vocabulary tokens, drawn either uniformly at random or weighted toward high-norm ("salient") tokens.

Every additional axis is then passed through Dikleştirme (native Gram-Schmidt) against the Pusula and against every other active axis before any Katkı is computed.

### Why a nonzero floor matters (unchanged since Phase III/IV)

In a classical critically damped system, the impulse response decays fully to zero. AkbasCore's Taban term means the direction is never fully released — the mathematical expression of the project's central design metaphor of an internal orientation rather than a gate that can be fully satisfied and then ignored. This remains true for every axis in Phase V, subject to each axis's own Katman-Derinlik Eğrisi.

## System Architecture

```
[ Tokenizer ] → [ Transformer Layer 0 ]──▶ AkbasCore hook ──▶ katkı computed (all active axes, in order)
                 [ Transformer Layer 1 ]──▶ AkbasCore hook ──▶ katkı computed (all active axes, in order)
                 [        ...          ]        ...
                 [ Transformer Layer N ]──▶ AkbasCore hook ──▶ katkı computed (all active axes, in order)
                 [ Layer N+1 ... final ]   (unmodified — no kernel activity)
                                  ↓
                          [ LM Head ] → [ Logits ] → [ Output tokens ]
```

- The kernel registers as a forward hook on the first N transformer layers (reference implementation: N = 20 of 28 total layers, 1.5B-parameter model). Deeper layers are left completely untouched.
- Two operating modes run on identical inputs for direct comparison: **Observer** (Vanilla — Katkı computed and logged, never written) and **Steered** (Katkı computed and written at every hooked layer). Running both back-to-back on the same prompt is a **Dual Run**.
- **Phase V addition — native multi-axis kernel.** Three compiled C++ entry points now implement the full mechanism: `akbas_orthogonalize` (one-time Dikleştirme of the axis matrix before generation begins), and `akbas_observe_multi` / `akbas_steer_multi` (the per-layer, per-axis loop described in [Mathematical Framework](#mathematical-framework)). All three remain host-side (CPU) C++, not `__global__` CUDA kernels.
- **Phase V addition — device safety.** Because the kernel is host-side C++, a CUDA tensor's `data_ptr` cannot be safely dereferenced from it. The Python harness now forces every tensor the kernel touches onto CPU before the call and moves the result back to the model's original device afterward, with an explicit `TORCH_CHECK` guard in the kernel that raises a clear error instead of segfaulting if this is ever bypassed. This makes the framework correct on both CPU-only environments (e.g. Termux/mobile) and GPU environments (e.g. Colab); see [Honest Limitations](#test-methodology-and-honest-limitations) for the associated performance cost on GPU.
- The reference implementation targets Qwen2.5-1.5B-Instruct (bfloat16, hidden dimension 1536) and earlier development used TinyLlama-1.1B-Chat. The method remains architecture-agnostic: any model exposing per-layer hidden states through a forward hook can be steered, provided every axis is reconstructed from that model's own embedding table.

## Development History — Phase I through Phase V

AkbasCore was developed iteratively and each phase is preserved in this repository for transparency, including the phases that did not work.

### Phase I — Linear Filter
*(unchanged from prior releases)*
The initial formulation treated alignment as a static linear combination, `P_t = (V₀ + Ω + Σφᵢ) × ε_t`, a philosophical mapping exercise rather than a mechanism grounded in model internals. It was not testable against real activations and is retained here as a documented negative result.

### Phase II — Resonance Field
*(unchanged from prior releases)*
Replaced the static formula with an undamped oscillator, `P_t = cos(θ) × sin(ωt + φ)`, introducing a real, computable cosine-similarity term for the first time. A pure sine wave never converges to a stable operating point — mathematically unstable, but the first step from ethics-as-filter to ethics-as-field.

### Phase III — Damped Kernel (Simulational Proof of Concept)
*(unchanged from prior releases)*
Introduced critical damping (ζ = 1) and the closed form used throughout this repository. Validated on 50 hand-constructed, manually-cosine-assigned scenarios — a deterministic simulation of the damping formula's stability, explicitly **not** live-model inference. This distinction is preserved deliberately as an example of correctly scoping a simulation result.

### Phase IV — Empirical Grounding and Live-Model Testing
*(unchanged from prior releases)*
Moved from hand-assigned vectors to a Pusula constructed from a real model's own embedding table, and from a simulation harness to a compiled C++ kernel hooked into live forward passes of Qwen2.5-1.5B-Instruct. This is the phase documented by the versioned kernel (AkbasCore 1.1 → 1.2), the Dual Run test protocol, and the numbered test-log series. Version 1.2 added the four live-adjustable motor controls (İvme, Sönüm, Zirve, Taban) exposed through a Gradio interface.

### Phase V — Native Multi-Axis Manifold Engine (AkbasCore 2.0 — DRA/MANIFOLD)

Phase IV's single fixed Pusula could not express two things the project's own testing surfaced as real gaps: (a) layer-varying semantic dynamics — early layers are dominated by syntax and surface form, deep layers by abstract intent and tone, and a single global direction/weight cannot treat these differently — and (b) independent, simultaneous control of more than one concept (e.g. a logic/coherence direction and a style direction) without one drowning out the other.

**A documented false start.** Before arriving at the Phase V architecture, an unversioned experimental branch (informal working name "Sector") attempted to address (a) by injecting synthetic control vectors constructed from non-embedding geometry — a Gaussian mask over dimension indices, a random unit vector — combined with the hidden state via rank-1 warp and axial-rotation-torque operations. This produced per-layer unbounded multiplicative growth (a projection term reaching roughly 30× its initial magnitude by the final hooked layer) and visibly corrupted output. It is documented here, per this repository's own stated practice of preserving negative results, because it directly motivated the Phase V design constraint below. The branch was never released and carries no version number in this repository's history.

**The Phase V design constraint.** Every steering direction, without exception, must be constructed from the target model's own real embedding table — the same requirement the Pusula has always satisfied. Phase V's contribution is a native architecture for combining *multiple* such directions correctly, entirely in the compiled kernel:

- **Native multi-axis Dikleştirme.** Additional axes are made orthogonal to the Pusula and to each other via in-kernel Gram-Schmidt before generation begins, giving null-space separation from the safety/logic direction as a mathematical consequence rather than a bolted-on projection step.
- **Per-axis independent Fren state.** Each axis's velocity-brake history is tracked separately, so oscillation feedback from one axis cannot leak into another's gain computation.
- **Katman-Derinlik Eğrisi.** Each additional axis activates over its own configurable layer window (Merkez, Genişlik), directly addressing gap (a) above — a coherence axis and a style axis can now be configured to dominate at different depths.
- **Kutup (Çekici/İtici) and the Rezonans↔Çekici physics blend.** Two gain physics are now available and continuously blendable: the original Phase III/IV Rezonans physics (amplifies existing alignment sign) and a new Çekici physics (corrects toward a target regardless of current sign, scaled by Açısal Hata). Repulsion (İtici) requires no separate kernel physics — it is implemented as Çekici attraction toward a negated axis vector.
- **Full-generation cumulative logging.** An earlier internal build of this phase logged only the initial prompt-processing pass (S > 1) rather than the actual decode steps, meaning diagnostic tables reflected almost none of the generated text's dynamics. This was identified and fixed before release: the logging buffer now accumulates per-layer statistics across every generated token (decode steps only, prefill excluded) and reports the running average, with an explicit token count (N) printed alongside every table.
- **CUDA device safety.** See [System Architecture](#system-architecture).
- **UI-exposed sampling temperature.** Added specifically to let an operator isolate genuine steering effects from ordinary sampling variance by setting temperature to 0 (greedy) for paired Dual Run comparisons.

This is the current and active phase of the project.

## Parameter Reference

### Active controls (adjustable per run, within calibrated safe bands)

| Parameter | Meaning | Default | Safe band | Since |
|---|---|---|---|---|
| İvme | Steering velocity — axis 0 (Pusula) master amplitude | 0.50 | 0.20 – 0.80 | Phase IV |
| Sönüm | Decay rate — how fast the push fades across layers (shared across all axes) | 0.30 | 0.10 – 0.60 | Phase IV |
| Zirve | Amplitude — initial peak push at early layers (shared across all axes) | 0.70 | 0.30 – 1.00 | Phase IV |
| Taban | Permanent floor — residual pressure at deep layers (shared across all axes) | 0.20 | 0.05 – 0.40 | Phase IV |
| Yönlendirme Fiziği | Rezonans↔Çekici gain-physics blend (0 = Rezonans, 1 = Çekici) | 1.00 | 0.00 – 1.00 | **Phase V** |
| Eksen Sayısı | Number of additional axes active this run (K = 1 + this value) | 0 | 0 – 3 | **Phase V** |
| Eksen Gücü (per axis) | Independent strength dial for a given additional axis | 0.50 | 0.10 – 1.00 | **Phase V** |
| Merkez (per axis) | Layer index where a given additional axis begins activating | 15 | 0 – 19 | **Phase V** |
| Genişlik (per axis) | Number of layers over which a given axis's activation ramps in | 6 | 1 – 15 | **Phase V** |
| Kutup (per axis) | Çekici (attractor) or İtici (repulsor) | Çekici | — | **Phase V** |
| Sıcaklık *(standard sampling parameter, not an original coinage — see note below)* | Output sampling randomness; 0 = deterministic/greedy | 0.65 | 0.00 – 1.50 | **Phase V** |

> Axis 0 (Pusula) is exempt from Merkez/Genişlik gating and remains at full strength at every hooked layer, exactly as in Phase III/IV.

### Embedded constants (fixed in the compiled kernel)

| Parameter | Meaning | Value |
|---|---|---|
| Oran | Base scale factor in the Katkı formula | 0.32 |
| Doyum | Saturation upper threshold | 0.75 |
| Karşıt | Counter (misalignment) threshold | −0.40 |
| Sapma | Uncertainty correction weight | 0.20 |
| Fren | Velocity brake coefficient | 0.30 |
| Sınır | Norm-proportional Katkı ceiling ratio | 0.045 (clamped to [0.04, 0.20]) |
| Bileşim | Ethics/logic blend ratio in Pusula construction | 0.40 / 0.60 |

### Axis-construction methods (standard techniques, Turkish-labeled for interface consistency — not original coinages)

| Interface label | Standard technique | Notes |
|---|---|---|
| Kelime Kümesi Ortalaması | Mean pooling over a word list's embeddings | Direct axis from a single concept list |
| Zıt Kavram Farkı | Difference-of-means | Same family as Zou et al. (2023) — see [Relationship to Prior Work](#relationship-to-prior-work) |
| Token Örneklemesi | Seeded (uniform or high-norm-weighted) token sampling | Mean embedding of a sampled token set |

## How to Run It

The kernel is distributed as a single self-contained script (see repository file list) that:

1. Installs required dependencies (ninja, gradio, bitsandbytes).
2. Compiles the C++ kernel inline via `torch.utils.cpp_extension`.
3. Loads the target model and constructs the Pusula (and any additional axes) from its embedding table.
4. Launches a Gradio interface exposing the motor sliders, the multi-axis configuration panel, the sampling-temperature control, and a Dual Run button.

**Requirements:** Python 3.7+, PyTorch, a Hugging Face `transformers`-compatible causal LM. No external dependencies beyond the standard scientific Python stack; runs on CPU (with correspondingly longer generation times) or GPU.

```
# In a fresh environment (e.g. Google Colab):
# 1. Paste the full kernel script into a single cell.
# 2. Run the cell — it will install dependencies, compile the kernel,
#    load the model, and print a Gradio share link.
# 3. Open the link, set the motor sliders and (optionally) configure
#    up to three additional axes, enter a prompt, and press "Dual Run"
#    to get paired Vanilla / Steered outputs plus full per-layer,
#    per-axis activation logs (averaged across the entire generation)
#    for both passes.
```

## Test Methodology and Honest Limitations

This section is a required part of the documentation, not a disclaimer appended for legal reasons.

**What the Dual Run protocol demonstrates.** Running Observer and Steered mode on an identical prompt, with identical sampling parameters, isolates the effect of the hidden-state intervention from all other sources of output variation. Differences between the two outputs are attributable to the Katkı being written into the hidden state — provided sampling temperature is controlled for (see below).

**What "sub-threshold" means and does not mean.** *(unchanged from Phase IV)* At default settings, individual per-layer Katkı values fall below the bfloat16 numeric resolution floor. This is a real, verifiable property of floating-point arithmetic — it is not evidence, by itself, that the intervention meaningfully changed model behavior. Behavioral change must be established independently by comparing actual outputs.

**What remains unproven — Phase IV items (unchanged):**
- The Pusula-to-hidden-state mapping has not been pre-trained or validated against a labeled ethical-contrast dataset; it is a fixed geometric construction, not a learned classifier.
- No formal ablation study across a standardized benchmark has been run to quantify capability trade-offs.
- Sample sizes in the published test-log series are small and single-operator; they demonstrate the mechanism operates as designed, not a statistically powered behavioral study.
- Phase III validation numbers are simulation-only and should never be cited as live-model results.

**What remains unproven — Phase V additions:**
- **GPU throughput cost.** The CPU-forced device-safety design (see [System Architecture](#system-architecture)) round-trips the hidden state to host memory at every hooked layer. This is negligible on CPU-only hardware but measurably reduces tokens/second on GPU relative to a hypothetical native `__global__` CUDA kernel, which has not yet been written. Reported multi-axis test logs in this phase were run on CPU; GPU throughput numbers have not yet been separately benchmarked and published.
- **Cross-axis independence is only exact along the orthogonalized directions.** Dikleştirme guarantees zero component *along* the Pusula direction for every additional axis, but does not guarantee zero *conceptual* correlation — if an additional axis's seed words are semantically close to the Pusula's seed words, empirical testing observed a small (order of 5%, single-prompt, non-exhaustive) indirect shift in the Pusula's own measured alignment when other axes were active versus inactive. This has not been systematically characterized across prompts or axis-word-list choices.
- **Multi-axis Dual Run sample size to date is very small** (single-digit test prompts, one operator) and has not been separated from sampling-temperature noise in most published logs; the temperature-zero control described above should be treated as the required protocol for any claim about Phase V axis effects going forward, not yet as something every existing Phase V log already applied.

**Where the raw logs live.** *(unchanged)* Every individual test run — full per-layer activation logs, vanilla/steered output pairs, and settings — is posted publicly and chronologically to r/TinyLlama_TITAN and r/TinyLlama_AGI as it happens; this README summarizes and organizes that record but does not replace it.

**Why this section exists.** *(unchanged)* A system that cannot state what it has not yet shown is not trustworthy on the things it has shown.

## Relationship to Prior Work

*(unchanged from prior releases, with one Phase V clarification appended)*

- **Representation Engineering** (Zou et al., 2023) — established that linear directions in a model's hidden-state space correlate with, and can be used to influence, high-level behaviors. AkbasCore's Pusula construction is a runtime, embedding-table-derived variant of this idea. Phase V's difference-of-means axis-construction option is the same technique family as this prior work, and is documented as such rather than as an original contribution (see the note under [Terminology](#terminology-canonical-reference)).
- **Activation Addition** (Turner et al., 2023) — demonstrated that adding a fixed vector to a single layer's residual stream produces consistent behavioral shifts. AkbasCore differs by applying a time-varying, closed-loop gain across many layers, and — as of Phase V — across multiple simultaneously active, mutually orthogonalized directions rather than one fixed-magnitude addition at one layer.
- **Constitutional AI** (Anthropic, 2022) — applies stated principles to shape behavior through training-time critique and revision. AkbasCore explores an analogous constitutional structure (the Terazi weights) expressed as a purely inference-time geometric direction.
- Critical damping is a standard result in classical control theory; its application here — as the gain envelope of a transformer steering kernel, decaying to a nonzero floor — remains the specific novel contribution documented under the name Damped Resonance Alignment. The Phase V multi-axis extension (native Gram-Schmidt combination of several such envelopes, with independent per-axis layer-depth gating and a continuous resonance/attractor physics blend) is the specific novel contribution of this release.

A note on naming: earlier drafts of this project used the working name "TITAN." That name has been retired in this repository to avoid any confusion with the unrelated, independently published memory-architecture paper "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2024). AkbasCore is the sole name used for this system going forward. Separately, an early Phase V experimental branch used the informal working name "Sector"; that name is retired for the same reason it is documented in [Phase V](#phase-v--native-multi-axis-manifold-engine-akbascore-20--dramanifold) — to prevent an abandoned, unreleased approach from being confused with the shipped architecture.

## Roadmap

- Empirical validation of the Pusula mapping against a labeled ethical-contrast dataset.
- Capability-preservation benchmarking (MMLU, HumanEval, or comparable) across a range of İvme/Zirve/Yönlendirme Fiziği settings.
- Adversarial and jailbreak-prompt evaluation of the cosine-alignment and Fren/Karşıt response.
- Multi-model validation beyond the current Qwen2.5-1.5B / TinyLlama-1.1B reference implementations.
- **(Phase V) Native `__global__` CUDA kernel** to eliminate the host↔device round-trip and enable full-speed GPU multi-axis steering.
- **(Phase V) Systematic cross-axis correlation study** — quantify, across a range of axis-word-list pairs, how much conceptual (non-orthogonal-direction) correlation between axes affects each axis's independently measured alignment.
- **(Phase V) Temperature-controlled Dual Run re-run of all published multi-axis logs**, to separate steering effects from sampling noise before drawing behavioral conclusions from them.
- Formal write-up and submission for peer review, once the above steps are complete.

## License and Attribution

### License Scope — What MIT Covers and What It Does Not

This project uses two separate licensing scopes, and they are not interchangeable:

✅ **Covered by the MIT License** — the software artifacts only:
- The C++ kernel source (`akbas_observe`, `akbas_steer`, `akbas_orthogonalize`, `akbas_observe_multi`, `akbas_steer_multi`, and related compiled extension code)
- The Python harness (model loading, hook registration, Dual Run orchestration, axis-candidate construction)
- The Gradio interface code
- Any other executable code file in this repository

You may use, copy, modify, and redistribute these code files under standard MIT terms, including commercially, provided the copyright notice is retained.

❌ **NOT covered by the MIT License** — the intellectual content of the method itself:
- The Damped Resonance Alignment (DRA) method and its governing equations
- The Pusula runtime-construction technique
- The native multi-axis Manifold extension: Dikleştirme, Kutup (Çekici/İtici), Rezonans↔Çekici physics blend (Yönlendirme Fiziği), Açısal Hata, and per-axis Katman-Derinlik Eğrisi (Merkez/Genişlik) gating
- The full terminology set defined in this document: Pusula, Katkı, Taban, Sönüm, Zirve, İvme, Terazi, Bileşim, Oran, Doyum, Karşıt, Sapma, Fren, Sınır, Eksen, Kutup, Çekici, İtici, Rezonans, Yönlendirme Fiziği, Açısal Hata, Merkez, Genişlik, Katman-Derinlik Eğrisi, Dikleştirme
- The parameter calibration values and safe-band ranges
- The critically-damped-with-nonzero-floor formulation described in [Mathematical Framework](#mathematical-framework)

These are original scientific and mathematical contributions, not software, and MIT licensing (which governs source code copying) does not apply to them. Reimplementing this method in a different programming language, under a different name, or as part of a different codebase does not remove the attribution requirement. Note: the standard, pre-existing techniques documented under [axis-construction methods](#axis-construction-methods-standard-techniques-turkish-labeled-for-interface-consistency--not-original-coinages) (word-set mean pooling, difference-of-means, token sampling, and sampling temperature) are explicitly excluded from this attribution requirement, as they are not original contributions of this project.

In short: you are free to fork, run, and build on the code. You are not free to present the DRA method, the Pusula construction technique, the multi-axis Manifold extension, or the terminology above as your own original contribution without attribution.

**Method, terminology, and mathematical framework:** the Damped Resonance Alignment method (Phase III/IV) and the native multi-axis Manifold extension (Phase V), the Pusula runtime-construction technique, and the complete original terminology defined in this document are original contributions first published in this repository by Akbaş, June 2026 (DRA) and September 2026 (Manifold/Phase V). These are documented here, with full mathematical specification, a public timestamped commit history, and a parallel public test-log record on r/TinyLlama_TITAN and r/TinyLlama_AGI, as the canonical reference for this method and terminology.

**Suggested citation:**

```
Akbaş. AkbasCore: Damped Resonance Alignment and the Native Multi-Axis
Manifold Extension — A Sub-Threshold Inference-Layer Steering Framework
for Transformer Models. GitHub repository, v2.0, September 2026.
https://github.com/ceceli33/titan-cognitive-core
```

Base models used in development remain under their own original licenses and are unmodified by this work:
- TinyLlama-1.1B-Chat-v1.0 — Apache 2.0
- Qwen2.5-1.5B-Instruct — Apache 2.0 (Alibaba Cloud, Qwen Team)

AkbasCore does not alter the weights of either base model. All steering occurs at inference time via forward hooks; the underlying models remain fully and separately licensed under their respective terms.
