# AkbasCore — Damped Resonance Alignment (DRA)

**A sub-threshold, inference-layer activation steering framework for transformer language models.**

**Author:** Akbaş
**Status:** Independent research pre-print — not yet peer-reviewed
**Version:** 1.2 (June 2026)
**License:** MIT (kernel code) — see [License and Attribution](#license-and-attribution) for terminology/method attribution terms
**Repository:** this repository is the canonical, timestamped source for AkbasCore, the Damped Resonance Alignment (DRA) method, the Pusula compass-construction technique, and all associated terminology defined below.

**Live test logs:** ongoing test runs, per-layer activation logs, and output comparisons are actively published and timestamped in real time on two Reddit communities — **r/TinyLlama_TITAN** and **r/TinyLlama_AGI** — prior to being consolidated here. Those threads are the day-to-day development record; this repository is the stable, versioned reference.

---

## Table of Contents

1. [What This Is](#what-this-is)
2. [Why This Exists](#why-this-exists)
3. [Core Idea in One Paragraph](#core-idea-in-one-paragraph)
4. [Terminology (Canonical Reference)](#terminology-canonical-reference)
5. [Mathematical Framework](#mathematical-framework)
6. [System Architecture](#system-architecture)
7. [Development History — Phase I through Phase IV](#development-history--phase-i-through-phase-iv)
8. [Parameter Reference](#parameter-reference)
9. [How to Run It](#how-to-run-it)
10. [Test Methodology and Honest Limitations](#test-methodology-and-honest-limitations)
11. [Relationship to Prior Work](#relationship-to-prior-work)
12. [Roadmap](#roadmap)
13. [License and Attribution](#license-and-attribution)

---

## What This Is

AkbasCore is a small, self-contained inference-time intervention layer that sits between a transformer language model's internal layers during the forward pass. It does not fine-tune, retrain, or modify any model weight. It does not alter the input prompt. It reads the hidden state at each of a model's early transformer layers, computes a small directional correction using a closed-form damped-oscillator equation, and — depending on operating mode — either logs that correction (Observer mode) or writes it into the hidden state (Steered mode).

The design goal is a form of alignment that is **geometric rather than linguistic**: instead of filtering tokens, blocklisting phrases, or training a classifier to gate outputs, AkbasCore defines a fixed direction in the model's own representation space (built from the model's own embedding table) and nudges the hidden state toward that direction by an amount that follows a critically damped decay profile across layers.

This repository is the complete, canonical record of that method: the mathematics, the terminology, the C++ kernel implementation, the parameter calibration, and the test logs produced while developing it.

## Why This Exists

Most production alignment techniques operate at one of two points: **during training** (RLHF, Constitutional AI, supervised fine-tuning) or **at the input/output boundary** (system prompts, moderation classifiers, output filters). Both are effective but come with trade-offs — training-time methods are computationally expensive and freeze behavior into the weights; boundary-time methods are comparatively easy to work around because they act on the finished token stream rather than the internal computation that produced it.

AkbasCore explores a third point: **the forward pass itself.** The hypothesis under test is whether a small, mathematically principled push on hidden-state geometry, applied consistently across a subset of layers, can measurably shift output character without weight modification and without the intervention needing to be large enough for standard instrumentation (cosine similarity at bfloat16 resolution) to register it as a discrete event.

This is presented as a hypothesis and an engineering experiment, not a proven alignment solution. The [Honest Limitations](#test-methodology-and-honest-limitations) section is a required part of this documentation, not an afterthought.

## Core Idea in One Paragraph

A single unit vector — the **Pusula** (compass) — is constructed at model-load time by averaging the model's own token embeddings for a curated set of ethical and logical anchor words, weighted by a fixed constitution. At each of the first *N* transformer layers during generation, the kernel measures the cosine similarity between the current hidden state and the Pusula, computes a scalar contribution (the **Katkı**) using a critically damped resonance formula that starts strong and decays to a small permanent floor, and — in Steered mode — adds that scalar, scaled along the Pusula direction, to the hidden state. The decay profile, floor, and overall gain are user-adjustable within calibrated safe bands; the underlying formula and compass-construction method are fixed architecture.

---

## Terminology (Canonical Reference)

All terms below are **original coinages by Akbaş**, introduced in this project, and are not translations of pre-existing terminology in the activation-steering or alignment literature. They are given in Turkish with fixed English glosses. Researchers extending or citing this work should use the canonical (diacritic-correct) spelling.

| Term | Canonical spelling | Gloss | Role |
|---|---|---|---|
| **Pusula** | Pusula | Compass Vector | The normalized unit direction in hidden-state space, built at runtime from the model's own embedding table, representing the target alignment direction. |
| **Katkı** | Katkı | Contribution | The scalar value computed at each layer and (in Steered mode) added to the hidden state along the Pusula direction. |
| **Taban** | Taban | Permanent Floor | The nonzero value the resonance gain decays *toward* rather than decaying to zero — a deliberate departure from the classical critically damped oscillator. |
| **Sönüm** | Sönüm | Decay Rate | Controls how quickly the resonance peak fades across layers. |
| **Zirve** | Zirve | Amplitude | The initial peak strength of the resonance push at the earliest steered layer. |
| **İvme** | İvme | Steering Velocity | The master amplitude dial; scales all Katkı values proportionally without altering the shape of the decay curve. |
| **Terazi** | Terazi | Constitution Weights | The fixed weighting given to each ethical category (harm avoidance, honesty, autonomy, fairness) when constructing the ethical component of the Pusula. |
| **Bileşim** | Bileşim | Blend Ratio | The mixing ratio between the ethical-embedding average and the logic-anchor-embedding average used to build the final Pusula. |
| **Oran** | Oran | Scale Factor | A fixed multiplier applied in the Katkı formula (embedded constant, not user-adjustable). |
| **Doyum** | Doyum | Saturation Threshold | The upper cosine-alignment threshold above which push is reduced, preventing over-steering of already well-aligned states. |
| **Karşıt** | Karşıt | Counter Threshold | The lower cosine-alignment threshold below which push is amplified, for strongly misaligned states. |
| **Sapma** | Sapma | Uncertainty Corrector | Adjusts the effective decay rate upward when alignment is ambiguous (cosine near zero). |
| **Fren** | Fren | Velocity Brake | A closed-loop correction that reduces push when alignment is improving layer-over-layer and increases it when alignment is worsening. |
| **Sınır** | Sınır | Max Katkı Ratio | A norm-proportional ceiling that bounds the Katkı magnitude relative to the hidden-state vector's own norm. |
| **Damped Resonance Alignment (DRA)** | — | — | The overall method name: applying a critically damped (ζ = 1) oscillator profile, decaying to a nonzero floor, as the per-layer gain envelope of an inference-time steering kernel. |

Two categories of parameter exist. **Active controls** (İvme, Sönüm, Zirve, Taban) are exposed to the operator within calibrated safe bands and can be adjusted per run. **Embedded constants** (Oran, Doyum, Karşıt, Sapma, Fren, Sınır) are compiled into the kernel and were fixed through iterative calibration; changing them requires recompilation.

---

## Mathematical Framework

### The governing equation

The per-layer gain follows a **critically damped oscillator** (damping ratio ζ = 1), the unique damping condition at which a disturbed system returns to equilibrium in the shortest possible time without oscillating past it:

```
P_t = cos(θ) × [ Zirve · e^(−Sönüm·t) · (1 + Sönüm·t) + Taban ]
```

Where `t` is the transformer layer index (0 through the last steered layer) and `cos(θ)` is the real-time cosine alignment between the current hidden state and the Pusula.

This is a deliberate departure from the textbook critically damped response, which decays to zero. Here the system decays to a **nonzero permanent floor (Taban)** — meaning even at the deepest steered layer, aligned hidden states continue to receive a small, continuous directional signal rather than the intervention switching off entirely.

### Step-by-step Katkı computation

At each layer and for each token position, the kernel performs seven steps:

**1. Alignment measurement.**
```
cos(θ) = (h · Pusula) / (‖h‖ · ‖Pusula‖)
```

**2. Uncertainty-corrected decay rate.** Alignment uncertainty (`unc = 1 − |cos(θ)|`) slightly speeds up decay for ambiguous states:
```
ω_eff = Sönüm + unc × Sapma
```

**3. Base layer gain** from the damped resonance envelope:
```
kb = Zirve · e^(−ω_eff·t) · (1 + ω_eff·t) + Taban
```

**4. Velocity-adjusted gain.** A closed-loop brake compares the current layer's alignment to the previous layer's:
```
dr = clamp(cos(θ)_t − cos(θ)_(t−1), −0.15, +0.15)

if dr > 0:  kv = kb × (1 − dr × Fren)      # alignment improving → ease off
if dr < 0:  kv = kb × (1 + |dr| × Fren)    # alignment worsening → push harder
kv = clamp(kv, 0.05, 1.0)
```

**5. Saturation factor**, preventing over-correction of already-aligned states and amplifying correction of strongly opposed states:
```
if cos(θ) > Doyum:   son = (1 − cos(θ)) / (1 − Doyum)
if cos(θ) < Karşıt:  son = 1.6
else:                son = 1.0
```

**6. Norm-proportional ceiling** on the maximum permissible contribution:
```
R_max = clamp(‖h‖ × Sınır, floor, ceiling)
```

**7. Final contribution**, clamped and applied along the Pusula direction:
```
k_t = clamp(İvme × cos(θ) × kv × Oran × son, −R_max, +R_max)

h[j] += k_t × Pusula[j]     for each hidden dimension j
```

### Constructing the Pusula

The compass vector is built once, at model-load time, directly from the target model's own embedding table — not from an external dataset or offline contrast-pair corpus.

```
ethics_vec = Σ [ Terazi_c × mean(embedding(seed_words_c)) ] / Σ Terazi_c   for each category c

logic_vec  = mean(embedding(logic_anchor_words))

Pusula = normalize( Bileşim_ethics × ethics_vec + Bileşim_logic × logic_vec )
```

Four constitutional categories (harm avoidance, honesty, autonomy respect, fairness) each contribute a weighted average of five seed-word embeddings. A separate set of fifteen logic-anchor words (e.g. *systematic, verifiable, deterministic, rigorous*) contributes a reasoning-oriented component. The two are blended and normalized to a unit vector — the direction every steered layer measures against and nudges toward.

### Why a nonzero floor matters

In a classical critically damped system, the impulse response decays fully to zero — the system "forgets" the initial disturbance. AkbasCore's Taban term means the direction is never fully released: even at maintenance/equilibrium layers, a small constant pressure toward the Pusula persists. This is the mathematical expression of the project's central design metaphor — an internal orientation rather than a gate that can be fully satisfied and then ignored.

---

## System Architecture

```
[ Tokenizer ] → [ Transformer Layer 0 ]──▶ AkbasCore hook ──▶ katkı computed
                 [ Transformer Layer 1 ]──▶ AkbasCore hook ──▶ katkı computed
                 [        ...          ]        ...
                 [ Transformer Layer N ]──▶ AkbasCore hook ──▶ katkı computed
                 [ Layer N+1 ... final ]   (unmodified — no kernel activity)
                                  ↓
                          [ LM Head ] → [ Logits ] → [ Output tokens ]
```

- The kernel registers as a **forward hook** on the first *N* transformer layers (in the reference implementation, N = 20 of 28 total layers for a 1.5B-parameter model). Deeper layers are left completely untouched.
- **Two operating modes** run on identical inputs for direct comparison:
  - **Observer mode** (Vanilla): the kernel computes the Katkı at every layer and logs it, but never writes to the hidden state. The output is the model's unmodified behavior.
  - **Steered mode**: the kernel computes and writes the Katkı to the hidden state at every hooked layer.
- Running both modes back-to-back on the same prompt is a **Dual Run**, producing two complete outputs and two full per-layer activation logs for side-by-side inspection.
- The kernel is implemented as a compiled C++ extension invoked from a PyTorch forward hook, keeping per-layer overhead low (on the order of a few percent of total generation time in reference testing).
- The reference implementation targets Qwen2.5-1.5B-Instruct (bfloat16, hidden dimension 1536) and earlier development used TinyLlama-1.1B-Chat. The method is architecture-agnostic: any model exposing per-layer hidden states through a forward hook can be steered, provided the Pusula is reconstructed from that model's own embedding table.

---

## Development History — Phase I through Phase IV

AkbasCore was developed iteratively and each phase is preserved in this repository for transparency, including the phases that did not work.

### Phase I — Linear Filter
The initial formulation treated alignment as a static linear combination:
```
P_t = (V₀ + Ω + Σφᵢ) × ε_t
```
with a scalar ethical constant V₀, an experience term Ω, randomized emotional fluctuation terms Σφᵢ, and an error-tolerance multiplier ε_t. This was a philosophical mapping exercise (Kantian fixed ethical anchor, Aristotelian experience, Damasian emotional weighting, Popperian falsifiability) rather than a mechanism grounded in model internals. It was not testable against real activations and is retained here as a documented negative result: **static, scalar filters do not have a mechanism for measuring or responding to a model's actual internal state.**

### Phase II — Resonance Field
The static formula was replaced with an undamped oscillator:
```
P_t = cos(θ) × sin(ωt + φ)
```
introducing, for the first time, a real cosine-similarity term `cos(θ)` between an input representation and a multi-dimensional V₀ vector — meaning alignment became, in principle, computable rather than asserted. However, a pure sine wave oscillates indefinitely; the system never converges to a stable operating point. This phase established the shift from **ethics as a filter to ethics as a field**, but was mathematically unstable.

### Phase III — Damped Kernel (Simulational Proof of Concept)
Phase III introduced critical damping (ζ = 1) to force convergence:
```
P_t = cos(θ) × [ A · e^(−ζωt) · sin(ωt + φ) + P∞ ]
```
and, at ζ = 1, the closed form used throughout this repository:
```
P_t = cos(θ) × [ A · e^(−ωt) · (1 + ωt) + P∞ ]
```
This phase also introduced the first multi-dimensional V₀ vector (five ethical dimensions instead of one scalar) and a reference PyTorch `nn.Module` implementation showing how such a layer could sit inside a transformer's residual stream.

**Important scope note:** the Phase III validation set (50 hand-constructed scenarios, categorized as aligned / neutral / misaligned) was a **deterministic mathematical simulation over manually assigned cosine values**, not live inference on a deployed model. It demonstrated the *stability and convergence properties* of the damping formula itself — that critical damping produces zero-overshoot convergence, and that the formula scales output proportionally rather than via binary gating — but did not yet demonstrate anything about a real model's hidden states. This distinction is preserved deliberately in this repository as an example of correctly scoping a simulation result.

### Phase IV — Empirical Grounding and Live-Model Testing
Phase IV moved from hand-assigned vectors to a Pusula constructed from a real model's own embedding table (Section: [Constructing the Pusula](#constructing-the-pusula)), and from a simulation harness to a compiled C++ kernel hooked into live forward passes of Qwen2.5-1.5B-Instruct. This is the phase documented by the versioned kernel (AkbasCore 1.1 → 1.2), the Dual Run test protocol, and the numbered test log series referenced throughout this repository. Version 1.2 added four live-adjustable motor controls (İvme, Sönüm, Zirve, Taban) exposed through a Gradio interface, replacing fixed defaults with a calibrated safe-band range per parameter.

This is the current and active phase of the project.

---

## Parameter Reference

### Active controls (adjustable per run, within calibrated safe bands)

| Parameter | Meaning | Default | Safe band |
|---|---|---|---|
| İvme | Steering velocity — master amplitude | 0.50 | 0.20 – 0.80 |
| Sönüm | Decay rate — how fast the push fades across layers | 0.30 | 0.10 – 0.60 |
| Zirve | Amplitude — initial peak push at early layers | 0.70 | 0.30 – 1.00 |
| Taban | Permanent floor — residual pressure at deep layers | 0.20 | 0.05 – 0.40 |

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

---

## How to Run It

The kernel is distributed as a single self-contained script (see repository file list) that:

1. Installs required dependencies (`ninja`, `gradio`, `bitsandbytes`).
2. Compiles the C++ kernel inline via `torch.utils.cpp_extension`.
3. Loads the target model and constructs the Pusula from its embedding table.
4. Launches a Gradio interface exposing the four active-control sliders and a Dual Run button.

**Requirements:** Python 3.7+, PyTorch, a Hugging Face `transformers`-compatible causal LM. No external dependencies beyond the standard scientific Python stack; runs on CPU (with correspondingly longer generation times) or GPU.

```bash
# In a fresh environment (e.g. Google Colab):
# 1. Paste the full kernel script into a single cell.
# 2. Run the cell — it will install dependencies, compile the kernel,
#    load the model, and print a Gradio share link.
# 3. Open the link, set the four motor sliders, enter a prompt,
#    and press "Dual Run" to get paired Vanilla / Steered outputs
#    plus full per-layer activation logs for both passes.
```

---

## Test Methodology and Honest Limitations

This section is a required part of the documentation, not a disclaimer appended for legal reasons.

**What the Dual Run protocol demonstrates.** Running Observer and Steered mode on an identical prompt, with identical sampling parameters (temperature, top-p, top-k, repetition penalty), isolates the effect of the hidden-state intervention from all other sources of output variation. Differences between the two outputs are attributable to the Katkı being written into the hidden state.

**What "sub-threshold" means and does not mean.** The kernel is designed so that, at default settings, individual per-layer Katkı values fall below the bfloat16 numeric resolution floor (~0.0078 per unit at typical hidden-state magnitudes). This means a naive layer-by-layer cosine-difference check will read as zero even though a value was, in fact, added to the tensor at full float32 precision before any downcast. This is a real, verifiable property of floating-point arithmetic — **it is not evidence, by itself, that the intervention meaningfully changed model behavior.** Behavioral change must be established independently by comparing actual outputs, which is why every test in this repository pairs numeric logs with full text outputs.

**What remains unproven.** As of this version:
- The `ethical_projector` / Pusula-to-hidden-state mapping has not been pre-trained or validated against a labeled ethical-contrast dataset; it is a fixed geometric construction, not a learned classifier.
- No formal ablation study across a standardized benchmark (e.g. MMLU, HumanEval) has been run to quantify whether steering strength trades off against general capability, though this is flagged as required future work.
- Sample sizes in the published test log series are small (single-digit to low-double-digit prompts per configuration) and were run by one operator. They demonstrate the mechanism operates as designed; they do not constitute a statistically powered behavioral study.
- The Phase III validation numbers are explicitly simulation-only (see [Phase III](#phase-iii--damped-kernel-simulational-proof-of-concept)) and should never be cited as live-model results.

**Where the raw logs live.** Every individual test run referenced in this project — full per-layer activation logs, vanilla/steered output pairs, and motor settings — is posted publicly and chronologically to **r/TinyLlama_TITAN** and **r/TinyLlama_AGI** as it happens, establishing a timestamped public record ahead of any consolidation into this repository. Those subreddits are the primary source for raw, unedited test data; this README summarizes and organizes that record but does not replace it.

**Why this section exists.** A system that cannot state what it has not yet shown is not trustworthy on the things it has shown. Anyone extending, critiquing, or attempting to replicate this work should start here.

---

## Relationship to Prior Work

AkbasCore builds on, and should be read alongside, the following prior work:

- **Representation Engineering** (Zou et al., 2023) — established that linear directions in a model's hidden-state space correlate with, and can be used to influence, high-level behaviors. AkbasCore's Pusula construction is a runtime, embedding-table-derived variant of this idea, rather than the offline contrast-pair method used in the original work.
- **Activation Addition** (Turner et al., 2023) — demonstrated that adding a fixed vector to a single layer's residual stream produces consistent behavioral shifts. AkbasCore differs by applying a *time-varying, closed-loop* gain across many layers (the damped resonance envelope and the Fren velocity brake) rather than a single fixed-magnitude addition at one layer.
- **Constitutional AI** (Anthropic, 2022) — applies a set of stated principles to shape model behavior through training-time critique and revision. AkbasCore explores an analogous constitutional structure (the Terazi weights) expressed as a geometric direction applied purely at inference time, with no training-time component.
- **Critical damping** is a standard result in classical control theory; its application here — as the gain envelope of a transformer steering kernel, decaying to a nonzero floor rather than to zero — is the specific novel contribution documented in this repository under the name Damped Resonance Alignment.

A note on naming: earlier drafts of this project used the working name "TITAN." That name has been retired in this repository to avoid any confusion with the unrelated, independently published memory-architecture paper "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2024), which shares no method, goal, or terminology with this project. **AkbasCore** is the sole name used for this system going forward.

---

## Roadmap

- **Empirical validation of the Pusula mapping** against a labeled ethical-contrast dataset, to move the compass direction from a fixed geometric construction toward an empirically validated one.
- **Capability-preservation benchmarking** (MMLU, HumanEval, or comparable) across a range of İvme/Zirve settings, to quantify any trade-off between steering strength and general task performance.
- **Adversarial and jailbreak-prompt evaluation**, to test whether cosine alignment predictably drops under adversarial pressure and whether the Fren/Karşıt mechanism responds as designed.
- **Multi-model validation** beyond the current Qwen2.5-1.5B / TinyLlama-1.1B reference implementations, to test whether the Pusula-construction method generalizes across architectures and parameter scales.
- **Formal write-up and submission for peer review**, once the above steps are complete.

---

## License and Attribution

### License Scope — What MIT Covers and What It Does Not

This project uses **two separate licensing scopes**, and they are not interchangeable:

**✅ Covered by the MIT License** — the software artifacts only:
- The C++ kernel source (`akbas_observe`, `akbas_steer` and related compiled extension code)
- The Python harness (model loading, hook registration, Dual Run orchestration)
- The Gradio interface code
- Any other executable code file in this repository

You may use, copy, modify, and redistribute these *code files* under standard MIT terms, including commercially, provided the copyright notice is retained.

**❌ NOT covered by the MIT License** — the intellectual content of the method itself:
- The Damped Resonance Alignment (DRA) method and its governing equations
- The Pusula runtime-construction technique
- The full terminology set defined in this document: Pusula, Katkı, Taban, Sönüm, Zirve, İvme, Terazi, Bileşim, Oran, Doyum, Karşıt, Sapma, Fren, Sınır
- The parameter calibration values and safe-band ranges
- The critically-damped-with-nonzero-floor formulation described in [Mathematical Framework](#mathematical-framework)

These are original scientific and mathematical contributions, not software, and MIT licensing (which governs source code copying) does not apply to them. **Reimplementing this method in a different programming language, under a different name, or as part of a different codebase does not remove the attribution requirement** — the requirement attaches to the method and terminology being used, not to the specific code file they first appeared in. Any use, extension, or independent reimplementation of the method or terminology listed above requires citation of this repository, regardless of what license the *surrounding* code is released under.

**In short:** you are free to fork, run, and build on the code. You are not free to present the DRA method, the Pusula construction technique, or the terminology above as your own original contribution without attribution — that content sits outside the MIT grant entirely.

**Method, terminology, and mathematical framework**: the Damped Resonance Alignment method, the Pusula runtime-construction technique, and the complete original terminology defined in this document are original contributions first published in this repository by **Akbaş**, June 2026. These are documented here, with full mathematical specification, a public timestamped commit history, and a parallel public test-log record on r/TinyLlama_TITAN and r/TinyLlama_AGI, as the canonical reference for this method and terminology.

Suggested citation:

```
Akbaş. AkbasCore: Damped Resonance Alignment — A Sub-Threshold Inference-Layer
Steering Framework for Transformer Models. GitHub repository, June 2026.
https://github.com/ceceli33/titan-cognitive-core
```

**Base models used in development** remain under their own original licenses and are unmodified by this work:
- TinyLlama-1.1B-Chat-v1.0 — Apache 2.0
- Qwen2.5-1.5B-Instruct — Apache 2.0 (Alibaba Cloud, Qwen Team)

AkbasCore does not alter the weights of either base model. All steering occurs at inference time via forward hooks; the underlying models remain fully and separately licensed under their respective terms.
