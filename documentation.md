AkbasCore: A Sub-Threshold Inference-Layer Steering Engine for Transformer Alignment via Damped Resonance Alignment

Author: Akbas
Repository: https://github.com/ceceli33/titan-cognitive-core
Status: Pre-print — not yet peer reviewed
Test Logs: All 76 tests published publicly at r/TinyLlama_TITAN on Reddit prior to this pre-print — timestamped public record
Version: 1.2
Date: June 2026


---


ABSTRACT


We present AkbasCore, a C++ runtime inference-layer steering engine that applies mathematically computed directional pressure to a transformer language model's hidden state at each transformer layer during inference. The intervention operates without retraining, without modifying model weights, and without altering the input prompt. A core design property of the system is that the pressure magnitude at each layer is designed to remain at or below the bfloat16 floating-point precision floor (approximately 0.0078 per unit); in the published default configuration, all recorded values fall within this range, though at higher active control settings individual layer values may approach or exceed this threshold. Standard cosine-similarity measurement instruments report zero change while structurally measurable differences in output quality are produced. We introduce an original mathematical framework called Damped Resonance Alignment (DRA) and a complete Turkish-named terminology system of fourteen original concepts. We report results across four consecutive domain tests — ethics (Test 72), mathematics (Test 73), philosophy (Test 75), and systems engineering (Test 76) — in which a consistent sub-threshold intervention produces structurally different outputs on a Qwen2.5-1.5B-Instruct model. All code is publicly available and fully replicable.


---


1. INTRODUCTION


Alignment of large language models has been approached primarily through training-time methods: Reinforcement Learning from Human Feedback (RLHF), Constitutional AI, and instruction fine-tuning. These methods modify model weights permanently and require significant computational resources. Inference-time methods, by contrast, intervene during the forward pass without touching the weights. Activation steering (Zou et al., 2023; Turner et al., 2023) demonstrated that adding a fixed vector to intermediate representations can steer model behavior. However, existing methods compute steering directions offline from behavioral contrast pairs, apply fixed scalar magnitudes, and do not incorporate real-time alignment feedback within the forward pass itself.

AkbasCore departs from this paradigm in three ways. First, the steering direction — which we call the pusula (compass vector) — is constructed at runtime from the model's own embedding table using a weighted constitutional framework, not from offline contrast pairs. Second, the steering magnitude follows a critically damped resonance profile that decays across transformer layers toward a nonzero permanent floor, implementing a closed-loop feedback mechanism at each layer. Third, the intervention magnitude is designed to remain at or below the bfloat16 precision floor in standard configuration, meaning the intervention is geometrically real but instrumentally invisible under typical measurement conditions.

We further introduce a complete original terminology in Turkish for all system concepts. These terms are canonical identifiers, not translations. Their use is required when citing or extending this work.

Note on the project name. Earlier test logs referred to the broader research project as TITAN. This name has been retired to avoid confusion with Google Research's independently published "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2024), which describes a fundamentally different architecture focused on long-term memory modules. The two systems share no conceptual, methodological, or terminological overlap. AkbasCore is the sole canonical name for all versions of this system.


---


2. RELATED WORK


Activation Steering. Zou et al. (2023) introduced Representation Engineering, demonstrating that linear directions in hidden state space correlate with model behaviors and can be used to steer outputs. Turner et al. (2023) showed that adding a fixed vector — termed an activation addition — to the residual stream at a single layer produces consistent behavioral changes. AkbasCore applies related principles but differs in direction construction method (runtime embedding average vs. contrast pairs), gain profile (critically damped resonance vs. fixed scalar), and the closed-loop per-layer feedback mechanism.

Constitutional AI. Anthropic (2022) applies ethical principles to model behavior through chain-of-thought critique during training. AkbasCore embeds analogous principles as a geometric direction in hidden state space applied at inference time, without any language-level processing.

Control Theory. The critically damped oscillator (zeta = 1) is a classical result in control engineering. Its impulse response A * e^(-omega*t) * (1 + omega*t) decays monotonically to zero without overshoot. AkbasCore applies this mathematical form to the per-layer gain profile of a transformer steering kernel, with the novel modification that decay targets a nonzero permanent floor rather than zero.

bfloat16 Precision. The bfloat16 floating-point format, standard for transformer inference, has a precision floor of approximately 0.0078 per unit at typical hidden state magnitudes. To our knowledge, deliberately designing an activation steering intervention to operate at or below this floor — such that standard instruments may not detect it under default conditions — is not a documented technique in the alignment literature.


---


3. METHODOLOGY


3.1 System Architecture

AkbasCore operates as a set of PyTorch forward hooks registered on the first N transformer layers. At each hooked layer, the C++ kernel receives the hidden state tensor, computes a scalar intervention value (the katkı), and adds this value scaled along the pusula direction to every token's hidden state. The model weights, tokenizer, and all other components are unchanged.

Two operating modes are supported. In Vanilla Observer mode, the kernel computes all values but does not write to the hidden state. All computed values are logged as delta-ref (theoretical contribution). In Steered mode, the kernel computes and writes. Running both modes on the same input in sequence is called a Dual Run, producing two complete outputs and two full activation logs for direct comparison.


3.2 The Pusula (Compass Vector)

The pusula is the target direction in the model's hidden state space. It is constructed once at initialization. For each of four constitutional categories (harm avoidance, honesty, autonomy, fairness), the embeddings of five seed words are retrieved from the model's own embedding table and averaged. These four category averages are combined as a weighted sum using the terazi (constitution weights):

```
ethics_vec  =  sum( W_c[i] * mean(emb[seeds_i]) )  /  sum( W_c[i] )
```

Separately, the embeddings of fifteen logic anchor words are averaged to form a logic vector. The pusula is then constructed as a weighted blend — the bileşim (blend ratio) — of the ethics and logic vectors, then normalized to unit length:

```
pusula  =  normalize( beta * ethics_vec  +  (1 - beta) * logic_vec )

beta = 0.40  (ethics weight)
1 - beta = 0.60  (logic weight)
```

The pusula encodes both ethical orientation and structured analytical reasoning. The 60% logic weighting ensures the compass remains geometrically proximate to technical domain representations, not only to ethical content.

Constitution weights (terazi):

```
harm avoidance    W = 0.9228    [safe, harmless, protective, secure, careful]
honesty           W = 0.9372    [honest, accurate, truthful, transparent, precise]
autonomy          W = 0.8788    [autonomous, respectful, unbiased, free, neutral]
fairness          W = 0.9196    [fair, just, equitable, balanced, impartial]
```

Logic anchors (15 words): logical, empirical, systematic, structured, verifiable, analyze, precise, deterministic, sequential, causal, rigorous, impossible, contradiction, identify, optimize.


3.3 The Katkı Formula (Contribution)

At each transformer layer t and for each token, the C++ kernel computes the katkı (contribution) k_t in seven steps.

Step 1 — Cosine alignment score. The real-time alignment between the hidden state h and the compass direction:

```
cos(theta)  =  dot(h, pusula)  /  ( norm(h) * norm(pusula) )
```

Step 2 — Effective decay rate. Uncertainty of alignment: unc = 1 - |cos(theta)|. The sapma (uncertainty corrector, delta = 0.20) adjusts the effective decay rate so that poorly aligned hidden states experience a slightly faster resonance decay:

```
omega_eff  =  omega  +  unc * delta
```

Step 3 — Layer gain kb from the damped resonance formula. This is the critically damped impulse response. At t = 0, kb = A + P_inf = 0.90. As t approaches infinity, kb approaches taban = 0.20:

```
kb  =  A * e^(-omega_eff * t) * (1 + omega_eff * t)  +  P_inf
```

Step 4 — Velocity-adjusted gain kv. The fren (velocity brake, phi = 0.30) creates closed-loop control: if alignment is improving (dr > 0), push is reduced; if worsening (dr < 0), push is increased. If cos(theta) > 0.80 and dr < 0, dr is first multiplied by phi before the adjustment:

```
dr  =  clamp( cos(theta)_t  -  cos(theta)_(t-1),  -0.15,  +0.15 )

if dr > 0:   kv  =  kb * (1 - dr * phi)        [improving: reduce push]
if dr < 0:   kv  =  kb * (1 + |dr| * phi)      [worsening: increase push]
kv  =  clamp(kv,  0.05,  1.0)
```

Step 5 — Saturation factor son. The doyum threshold (Theta_max = 0.75) prevents over-pushing already well-aligned hidden states; son reaches 0.0 at cos(theta) = 1.0. The karşıt threshold (Theta_min = -0.40) amplifies push by 60% on strongly misaligned hidden states:

```
if cos(theta) > Theta_max:   son  =  (1 - cos(theta)) / (1 - Theta_max)
if cos(theta) < Theta_min:   son  =  1.6
else:                         son  =  1.0
```

Step 6 — Max katkı ceiling. The sınır (R_max) is norm-proportional. At norm = 12: clamp(12 * 0.045, 0.04, 0.20) = 0.20. At norm = 0.5: clamp(0.0225, 0.04, 0.20) = 0.04 (floor active):

```
R_max  =  clamp( norm(h) * 0.045,   0.04,   0.20 )
```

Step 7 — Final katkı computation and application. The oran (scale factor, lambda = 0.32) and ivme (steering velocity, v0 = 0.50) together scale the final value:

```
k_t  =  clamp( v0 * cos(theta) * kv * lambda * son,   -R_max,   +R_max )

h[j]  +=  k_t * pusula[j]     for each dimension j in [0 .. D-1]

ivme v0 = 0.50  |  oran lambda = 0.32  |  D = 1536 (Qwen2.5-1.5B)
```


3.4 Dual Pass Architecture

Pass A — Vanilla Observer: the kernel computes k_t at each layer but does not write to the hidden state. Values are logged as delta-ref. The output is the unmodified model response.

Pass B — Steered: the kernel computes and writes k_t at each layer. The output is the steered model response.

The Dcos (delta cosine) metric reports the difference in cos(theta) between the two passes at each layer. In the published default configuration, Dcos = 0.0000 at every layer to four decimal places, because each individual katkı (maximum +0.003864 at L=1 in default settings) remains at or below the bfloat16 resolution floor of approximately 0.0078. The cumulative sum across 20 layers is +0.034953. This is not a measurement failure; it is a design property of the default configuration.


3.5 Parameter Taxonomy

Architectural terms define what the system is. They are not adjustable and constitute the system's identity.

Active controls (ayar) are exposed in the user interface. Each has a calibrated safe band: wide enough to produce measurable output differences, narrow enough not to destabilize the system.

Embedded constants (motor) are compiled into the C++ kernel. Modifying them requires recompilation. They were established through empirical calibration across 76 published tests.

Active controls with defaults and safe bands:

```
ivme   (v0)      Steering Velocity     default 0.50    safe band [0.20, 0.80]
sönüm  (omega)   Decay Rate            default 0.30    safe band [0.10, 0.60]
zirve  (A)       Amplitude             default 0.70    safe band [0.30, 1.00]
taban  (P_inf)   Permanent Floor       default 0.20    safe band [0.05, 0.40]
```

Embedded constants:

```
oran    (lambda)     Scale Factor                  0.32
doyum   (Theta_max)  Saturation Upper Threshold    0.75
karşıt  (Theta_min)  Counter Threshold            -0.40
sapma   (delta)      Uncertainty Corrector         0.20
fren    (phi)        Velocity Brake                0.30
sınır   (R_max)      Max Katkı Ratio   0.045  (floor 0.04, ceiling 0.20)
```

Intermediate variables (defined within the formula, not exposed as parameters):

```
kb          base layer gain before velocity adjustment
kv          velocity-adjusted gain after fren correction
son         saturation factor combining doyum and karsit thresholds
dr          rate of change of cos(theta) between adjacent layers
unc         uncertainty term: 1 - |cos(theta)|
omega_eff   effective decay rate after sapma correction
norm(h)     L2 norm of the hidden state vector
prev_cos    cos(theta) from the previous layer (reset per inference call)
```

Log and measurement terms:

```
delta-ref   theoretical katkı in Vanilla Observer mode — never applied
Dcos        difference in cos(theta) between vanilla and steered passes per layer
log_buf     [20 x 4] tensor storing cos(theta), kb, kv, katkı for layers 0-19
drift       cos(theta) change from L0 to L19
peak push   maximum katkı, occurring at L=1 (+0.003864 in default configuration)
equilibrium maintenance mode entered from approximately L=15 onward
```


---


4. EXPERIMENTS


4.1 Setup

Primary model: Qwen/Qwen2.5-1.5B-Instruct, bfloat16, CPU inference (Tests 60-76). Development model: TinyLlama/TinyLlama-1.1B-Chat-v1.0, bfloat16, CPU inference (Tests 1-59). Steering applied to layers 0 through 19 of 28. Generation parameters: temperature 0.65, top-p 0.90, top-k 50, repetition penalty 1.15. All tests use the Dual Run configuration. All test results were published chronologically and publicly at r/TinyLlama_TITAN on Reddit, establishing a timestamped record of each run prior to this pre-print.


4.2 Cross-Domain Stability (Tests 72, 73, 75, 76)

The following cosine alignment values were recorded at L=0 and L=19 across four consecutive tests spanning entirely different subject domains:

```
Test 72  ethics                cos(theta) L0=+0.0134   L19=+0.0343   drift=+0.0209
Test 73  mathematics           cos(theta) L0=+0.0134   L19=+0.0343   drift=+0.0209
Test 75  philosophy            cos(theta) L0=+0.0134   L19=+0.0343   drift=+0.0209
Test 76  systems engineering   cos(theta) L0=+0.0134   L19=+0.0343   drift=+0.0209
```

The pusula produces an identical geometric signature across all four domains. We interpret this as evidence that the ethical-logical direction encoded in the pusula is a stable geometric property of the model's hidden state space, not a domain-specific artifact.


4.3 Test 76 — Instruction-Following Under Constrained Task

A three-part systems architecture problem was posed to both passes simultaneously. The task required: (1) selecting one specific algorithm from a stated set of thirty, (2) comparing it step-by-step against alternatives with architectural reasoning, and (3) implementing it in Python.

Vanilla response: invented a generic approach not drawn from the stated set of thirty; omitted the comparative analysis entirely; produced placeholder Python code with pass statements and no real imports.

Steered response: named TFO (Traffic Flow Optimization) from the known solution space; compared it against FIFO and Priority Queue approaches with specific architectural reasoning; produced a working scaffold with heapq and deque imports and a functional entry point.

Output token counts: Vanilla 721, Steered 757 (+5.0%). The difference is not in length but in structural compliance with the three-part task specification.


4.4 Sub-Threshold Verification

In the published default configuration, the maximum recorded single-layer katkı is +0.003864 at L=1. The bfloat16 resolution floor is approximately 0.0078. The ratio is 0.003864 / 0.0078 = 0.496. Under these settings, all recorded katkı values remain at or below the bfloat16 floor, and Dcos = 0.0000 at every layer. At higher active control settings, individual values may approach or exceed this threshold. The cumulative katkı across 20 layers in the default configuration is +0.034953, sufficient to produce structurally different outputs.


---


5. TERMINOLOGY REFERENCE


All Turkish names are canonical identifiers — original coinages by Akbas, not translations of existing terms. Researchers and developers who use, extend, or cite this system must use these names as given, including Turkish characters. ASCII approximations (katki, sonum, karsit, sinir, bilisim) may appear in code variable names due to identifier constraints but are not the canonical forms.

pusula — Compass Vector — C_vec — The normalized unit vector in the model's hidden state space, constructed at runtime from the model's own embedding table, encoding the target direction for steering.

katkı — Contribution — k_t — The scalar value computed by the C++ kernel and written to the hidden state at each transformer layer along the pusula direction. In Vanilla Observer mode it is computed but not applied (logged as delta-ref). Original coinage with no equivalent in prior AI/ML literature.

ivme — Steering Velocity — v0 — Master amplitude dial. Scales all katkı values proportionally without changing the resonance profile shape. Highest individual sensitivity among the four active controls. Safe band [0.20, 0.80]. Default 0.50.

sönüm — Decay Rate — omega — Controls how quickly the resonance peak fades across transformer layers. Lower values spread the push into deeper layers; higher values concentrate it in early layers. Equilibrium (maintenance mode) reached at approximately L=15 with the default value. Safe band [0.10, 0.60]. Default 0.30.

zirve — Amplitude — A — Initial amplitude of the resonance push at early layers. At t=0, kb = zirve + taban = 0.90. Safe band [0.30, 1.00]. Default 0.70.

taban — Permanent Floor — P_inf — Minimum continuous alignment pressure persisting after the exponential resonance term has decayed. Unlike classical critically damped systems which decay to zero, AkbasCore decays to this nonzero floor. Safe band [0.05, 0.40]. Default 0.20.

terazi — Constitution Weights — W_c — Four-category weighted ethical framework defining the ethical component of the pusula direction.

bileşim — Blend Ratio — beta — Ratio combining ethical embeddings and logic anchor embeddings in pusula construction. Default 0.40 ethics / 0.60 logic.

oran — Scale Factor — lambda — Base multiplier in the katkı formula (0.32).

doyum — Saturation Upper Threshold — Theta_max — When cos(theta) exceeds 0.75, the saturation factor son decreases proportionally, reaching 0.0 at cos(theta) = 1.0.

karşıt — Counter Threshold — Theta_min — When cos(theta) falls below -0.40, son = 1.6, amplifying katkı by 60%.

sapma — Uncertainty Corrector — delta — Adjusts the effective decay rate based on alignment uncertainty. Maximum correction +0.20 when cos(theta) = 0 (0.20).

fren — Velocity Brake — phi — Closed-loop gain coefficient. Reduces push when alignment is improving; increases push when worsening (0.30).

sınır — Max Katkı Ratio — R_max — Norm-proportional ceiling on katkı magnitude. Formula: clamp(norm * 0.045, 0.04, 0.20).

Damped Resonance Alignment (DRA) — The specific application of critical damping (zeta = 1) to define the per-layer decay profile of an inference-time transformer steering kernel, with a nonzero permanent floor replacing the classical zero equilibrium. No prior equivalent in the alignment literature. Original contribution of Akbas (2026).


---


6. DISCUSSION


6.1 The Permanent Floor as an Alignment Design Choice

Classical critically damped systems decay to zero. The choice to decay to a nonzero taban (P_inf = 0.20) is deliberate: even at layer 19, every token's hidden state receives a low-level push toward the compass direction. This ensures that deeply processed, already well-aligned representations continue to receive a directional signal. Because the magnitudes are at or below the bfloat16 floor in default configuration, this constitutes continuous geometric orientation rather than a forceful intervention.


6.2 Why Turkish Names

The Turkish names are not stylistic choices. They serve as unambiguous identifiers that cannot be confused with terms already present in the literature. When a researcher writes "katkı," there is exactly one system in the literature that uses this term in this sense. The names must be used with Turkish characters (katkı, sönüm, karşıt, sınır, bileşim) as these are the canonical forms.


6.3 Limitations

The pusula is a fixed geometric direction that does not adapt to prompt content. On inputs where the model's internal representations are geometrically distant from the constitutional seed word cluster, cos(theta) approaches zero and katkı approaches zero as well; the system reduces to taban-level floor pressure. The system steers; it does not constrain. Runtime overhead on CPU for Qwen2.5-1.5B: approximately 9 seconds over a full generation of 700+ tokens, approximately 1.8% overhead relative to the vanilla pass.


6.4 On the Name TITAN

Earlier test posts (Tests 1 through 76, r/TinyLlama_TITAN) referred to the broader research project as TITAN. This name is retired. Google Research published "Titans: Learning to Memorize at Test Time" (Behrouz et al., 2024) in December 2024, describing a memory architecture for transformers. The two systems are entirely unrelated in method, goal, and terminology. TITAN is not used in this or any future publication. The system is AkbasCore in all contexts.


---


7. CONCLUSION


AkbasCore demonstrates that sub-threshold directional pressure applied to transformer hidden states across twenty layers produces structurally different model outputs across four consecutive domain tests. The intervention remains at or below the bfloat16 precision floor in standard configuration while producing measurable differences in instruction-following precision, comparative reasoning quality, and code implementation completeness. The system introduces an original mathematical framework (Damped Resonance Alignment), an original runtime compass construction method (pusula), and a complete Turkish-named terminology of fourteen canonical concepts. All components are publicly available, replicable from a single Colab cell, and verified across seventy-six published tests on two model families.


---


REFERENCES


Anthropic. (2022). Constitutional AI: Harmlessness from AI feedback. arXiv:2212.08073.

Behrouz, A., Zheng, P., Mirrokni, V., & Karbasi, A. (2024). Titans: Learning to memorize at test time. arXiv:2501.00663.

Damasio, A. R. (1994). Descartes' Error: Emotion, Reason, and the Human Brain. Putnam.

Kant, I. (1785). Groundwork of the Metaphysics of Morals.

Popper, K. (1959). The Logic of Scientific Discovery. Hutchinson.

Turner, A., Thiergart, L., Udell, G., Leech, G., Mini, U., & MacDiarmid, M. (2023). Activation addition: Steering language models without optimization. arXiv:2308.10248.

Zou, A., Phan, L., Chen, S., Campbell, J., Guo, B., Bhatt, R., & Hendrycks, D. (2023). Representation engineering: A top-down approach to AI transparency. arXiv:2310.01405.


---


LICENSE AND MODEL INFORMATION


AkbasCore, the Damped Resonance Alignment method, and all associated terminology are original scientific contributions derived by Akbas in 2026. Any academic or commercial use of this methodology requires citation of this documentation.

AkbasCore 1.2 — Kernel code and implementation
License: MIT
Copyright (c) Akbas, June 2026

The MIT License covers the AkbasCore kernel code and its implementation only.

The Damped Resonance Alignment method, the constitutional compass construction approach, and all original terminology (pusula, katkı, taban, sönüm, zirve, ivme, terazi, bileşim, oran, doyum, karşıt, sapma, fren, sınır) are original scientific contributions of the author and are not covered by the MIT License. Any use, implementation, or derivative work based on these contributions — including independent reimplementations — requires citation: Akbas, AkbasCore v1.2, June 2026, https://github.com/ceceli33/titan-cognitive-core.

Base models used in this research:

```
TinyLlama/TinyLlama-1.1B-Chat-v1.0   Tests 1-59    License: Apache 2.0
  Author: Zhang Peiyuan et al.
  https://github.com/jzhang38/TinyLlama

Qwen/Qwen2.5-1.5B-Instruct            Tests 60-76   License: Apache 2.0
  Author: Alibaba Cloud — Qwen Team
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
```

AkbasCore does not modify the weights of either model. It applies inference-time activation steering only. Both base models remain fully under their respective Apache 2.0 licenses.

Repository: https://github.com/ceceli33/titan-cognitive-core

Pre-print — AkbasCore v1.2 — (c) Akbas — June 2026
