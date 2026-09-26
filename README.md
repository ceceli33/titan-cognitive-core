# AkbasCore — SEASC

**Standardized Evidential Activation-Steering Core**

> **A model-internal steering motor that synthesizes its own layer-local compasses, applies measured physical intervention, stops deliberately, and observes what the transformer does next.**

## KIZILELMA LIVE X-RAY

### SEE THE INTERVENTION, NOT JUST THE OUTPUT.

### [OPEN KIZILELMA LIVE X-RAY](./AkbasCore_SEASC_KIZILELMA_Live_X_Ray)

**SYNTHESIZE → INJECT → SHAPE → STOP → OBSERVE**

**Author:** Mustafa Akbaş  
**Status:** Independent research / experimental reference implementation  
**Current research line:** AkbasCore SEASC — September 2026  
**Reference model:** Qwen/Qwen2.5-7B-Instruct  
**Reference hardware:** NVIDIA A100  
**Model weights modified during steering:** No  
**Fine-tuning required:** No  
**LoRA required:** No  
**External steering vector required:** No  

> **AkbasCore is an experimental Reference Core and research methodology, not a claim of a universal production alignment system.**

---

# The Core Idea

AkbasCore does not begin with a stored steering vector.

It does not load a behavioral adapter.

It does not fine-tune the model.

It does not modify the model weights.

Instead, the current AkbasCore motor uses the transformer's **own internal activation geometry** to synthesize the directions that will subsequently be used to intervene in that same model.

For the reference Qwen2.5-7B-Instruct configuration:

```text
20 independently synthesized
layer-local compasses

20 × 3,584 dimensions

71,680 directional values
```

are constructed for the active intervention region.

The motor combines:

- model-internal compass synthesis;
- frozen-norm physical dose;
- critically damped depth shaping;
- direct CUDA hidden-state injection;
- explicit intervention cutoff;
- motor-OFF downstream observation.

This complete mechanism is the current AkbasCore SEASC architecture.

---

# The AkbasCore Motor

The current motor follows this sequence:

```text
MATCHED SEMANTIC CONTRASTS

        ↓

KARŞITLIK OCAĞI
Contrast Forge

        ↓

MODEL-INTERNAL ACTIVATIONS

        ↓

SENTETİK PUSULA ÜRETİMİ
Synthetic Compass Synthesis

        ↓

20 KATMAN PUSULASI
20 Layer-Local Compasses

        ↓

PUSULA DEMETİ
Compass Bundle
20 × 3,584

        ↓

DRA DOZ ZARFI

        ↓

DONDURULMUŞ NORM ÖLÇÜMÜ

        ↓

NORMA BAĞLI PUSULA ENJEKSİYONU

        ↓

L0 → L19
SEASC ON

        ↓

KESİM SINIRI

        ↓

MOTOR OFF

        ↓

L20 → L27
SESSİZ KUYRUK

        ↓

İZ TAŞINIMI

        ↓

KIZILELMA
LIVE X-RAY
```

The twenty synthetic layer-local directions are:

```text
L00 → A0
L01 → A1
L02 → A2
L03 → A3
...
L18 → A18
L19 → A19
```

Each direction contains **3,584 real values**.

The motor therefore separates six questions:

1. What semantic contrast are we measuring?
2. What direction does that contrast have inside each layer?
3. How large should the physical intervention be?
4. How should that dose vary through depth?
5. Where should active intervention stop?
6. What happens after the motor stops?

---

# 1. Karşıtlık Ocağı — Contrast Forge

**Karşıtlık Ocağı** is the AkbasCore stage that converts matched semantic oppositions into measurable internal activation contrasts.

Reference semantic axes include:

```text
ABOVE  ↔ BELOW
BEFORE ↔ AFTER
INSIDE ↔ OUTSIDE
OPEN   ↔ CLOSED
CAUSE  ↔ EFFECT
```

Matched positive and negative carrier sentences are passed through the same model.

For every target layer \(L\), AkbasCore records:

\[
h_L^{POS}
\]

and:

\[
h_L^{NEG}
\]

The mean activation of each side is then computed:

\[
\mu_L^{POS}
=
\operatorname{mean}(h_L^{POS})
\]

\[
\mu_L^{NEG}
=
\operatorname{mean}(h_L^{NEG})
\]

Karşıtlık Ocağı does not steer the model.

Its purpose is to expose a measurable semantic contrast inside the model's own activation geometry.

---

# 2. Sentetik Pusula Üretimi — Synthetic Compass Synthesis

This is one of the central mechanisms of the current AkbasCore motor.

**AkbasCore does not require a stored or externally supplied steering vector.**

Instead, before active steering begins, the model is used to synthesize its own layer-local intervention directions from matched internal activation contrasts.

For each layer:

\[
v_L^{sentetik}
=
\mu_L^{POS}
-
\mu_L^{NEG}
\]

The synthetic contrast is normalized:

\[
A_L
=
\frac{v_L^{sentetik}}
{\|v_L^{sentetik}\|_2}
\]

This operation is performed independently at every active layer:

\[
A_0,A_1,A_2,\ldots,A_{19}
\]

For Qwen2.5-7B-Instruct:

\[
A_L \in \mathbb{R}^{3584}
\]

One synthesis cycle therefore produces:

```text
20 independent synthetic vectors

× 3,584 dimensions

= 71,680 directional values
```

before active steering begins.

The important distinction is:

> **The intervention directions are synthesized from the model's own layerwise activation geometry. They are not imported from an external steering-vector library.**

AkbasCore also does not assume that one semantic direction remains unchanged through transformer depth.

Instead:

```text
Layer 0  gets A0
Layer 1  gets A1
Layer 2  gets A2
...
Layer 19 gets A19
```

Each active layer receives its own synthetic direction.

---

# 3. Katman Pusulası — Layer-Local Compass

Each independently synthesized unit direction is called:

**Katman Pusulası — Layer-Local Compass**

For layer \(L\):

\[
A_L \in \mathbb{R}^{H}
\]

with:

\[
\|A_L\|_2=1
\]

The current motor does not assume:

\[
A_0=A_1=\cdots=A_{19}
\]

Semantic geometry is allowed to change through transformer depth.

---

# 4. Pusula Demeti — Compass Bundle

The complete collection:

\[
\mathcal{A}
=
\{A_0,A_1,\ldots,A_{19}\}
\]

is called:

**Pusula Demeti — Compass Bundle**

For the reference model:

```text
20 Layer-Local Compasses
×
3,584 dimensions
=
71,680 directional values
```

The entire Pusula Demeti is synthesized from the model's own activations.

---

# Scientific Context

Contrastive activation differences and steering-vector extraction have prior literature.

AkbasCore therefore does **not** claim that subtracting positive and negative activations, by itself, is a newly invented mathematical operation.

The AkbasCore architecture being investigated is the integrated mechanism:

```text
Layer-local model-internal
compass synthesis

        +

Frozen-norm physical dose

        +

DRA depth shaping

        +

Multi-layer CUDA intervention

        +

Explicit motor cutoff

        +

Motor-OFF downstream X-Ray

        +

Transport measurement
```

The project distinguishes its architectural mechanisms and experimental methodology from the standard mathematical operations used to implement them.

---

# 5. Pusula Dökümhanesi — Compass Foundry

**Pusula Dökümhanesi** is the umbrella term for AkbasCore direction-construction mechanisms.

The current direct synthesis path is:

```text
Pusula Dökümhanesi

        ↓

Karşıtlık Ocağı

        ↓

Sentetik Pusula Üretimi

        ↓

Katman Pusulaları

        ↓

Pusula Demeti
```

Historical Foundry research also includes:

- Banka Pusulası;
- Subspace-Fisher Reference;
- Artık Yön;
- Bileşik Pusula.

These remain part of the research lineage but are not required for the current direct KIZILELMA synthesis path.

---

# 6. Dondurulmuş Norm — Frozen Norm

Once a layer direction has been synthesized, AkbasCore determines how strongly to intervene.

Before injection, the motor measures:

\[
n_L
=
\|h_L\|_{\mathrm{pre}}
\]

AkbasCore calls this:

**Dondurulmuş Norm — Frozen Norm**

The norm is captured before intervention modifies the hidden state.

The intervention therefore cannot increase the reference norm from which its own requested magnitude is calculated.

---

# 7. Fiziksel Doz — Physical Dose

AkbasCore defines intervention magnitude relative to the hidden state being modified.

For relative layer dose \(\rho_L\):

\[
\Delta h_L
=
\rho_L
\|h_L\|_{\mathrm{pre}}
A_L
\]

Physical Dose is:

\[
D_L
=
\frac{\|\Delta h_L\|_2}
{\|h_L\|_{\mathrm{pre}}}
\]

Because:

\[
\|A_L\|_2=1
\]

the requested relative displacement satisfies:

\[
D_L \approx \rho_L
\]

up to numerical implementation precision.

A requested 10% intervention therefore means approximately:

> **a displacement equal to 10% of the pre-intervention hidden-state norm**

rather than an arbitrary uncalibrated additive coefficient.

---

# 8. DRA Zarfı — DRA Envelope

Physical dose is shaped through transformer depth.

AkbasCore retains the critically damped profile developed in the DRA research line:

\[
kb(L)
=
\mathrm{Zirve}
e^{-\mathrm{Sönüm}L}
(1+\mathrm{Sönüm}L)
+
\mathrm{Taban}
\]

The profile is normalized:

\[
E(L)
=
\frac{kb(L)}{kb(0)}
\]

therefore:

\[
E(0)=1
\]

and:

\[
\rho_L
=
\mathrm{İvme}\,E(L)
\]

Current locked constants:

| Turkish term | English gloss | Value |
|---|---|---:|
| **İvme** | Initial Physical Steering Dose | 0.10 |
| **Sönüm** | Decay Rate | 0.30 |
| **Zirve** | Envelope Amplitude | 0.70 |
| **Taban** | Non-Zero Floor | 0.20 |

Reference dose schedule:

```text
L00 : 10.000%
L01 :  9.713%
L02 :  9.052%
L03 :  8.230%
L04 :  7.376%
L05 :  6.561%
L06 :  5.822%
L07 :  5.175%
L08 :  4.621%
L09 :  4.156%
L10 :  3.771%
L11 :  3.456%
L12 :  3.200%
L13 :  2.994%
L14 :  2.829%
L15 :  2.697%
L16 :  2.593%
L17 :  2.511%
L18 :  2.447%
L19 :  2.397%
```

The locked base envelope has:

```text
RSS(ρ) ≈ 0.250235055
```

The DRA Zarfı controls depth.

It does not determine semantic direction.

---

# 9. Norma Bağlı Pusula Enjeksiyonu

## Norm-Anchored Compass Injection

This is the point where the synthetic direction, hidden-state magnitude and physical dose meet.

For each active layer:

\[
\boxed{
h'_L
=
h_L
+
\rho_L
\|h_L\|_{\mathrm{pre}}
A_L
}
\]

The operation combines:

```text
Katman Pusulası

        +

Dondurulmuş Norm

        +

Fiziksel Doz

        +

DRA Zarfı
```

into one direct hidden-state intervention.

The reference implementation performs this operation through an inline CUDA kernel.

No optimizer is involved.

No gradient update is required.

No model weight is modified.

---

# The Complete Runtime Motor

The current AkbasCore runtime can be summarized as:

```text
SYNTHESIZE DIRECTION

        ↓

MEASURE HIDDEN-STATE NORM

        ↓

COMPUTE PHYSICAL DOSE

        ↓

SHAPE DOSE THROUGH DEPTH

        ↓

INJECT LAYER-LOCAL PUSULA

        ↓

REPEAT THROUGH L0-L19

        ↓

STOP

        ↓

OBSERVE L20-L27
```

Or mathematically:

\[
\text{AkbasCore}
=
\text{Synthetic Compass Synthesis}
+
\text{SEASC}
+
\text{DRA Envelope}
+
\text{Intervention Cutoff}
+
\text{Trace Observation}
\]

---

# 10. Kesim Sınırı — Intervention Cutoff

AkbasCore deliberately separates active intervention from downstream observation.

```text
L0 → L19
SEASC ACTIVE
INJECTION ON

----------------

KESİM SINIRI
INTERVENTION CUTOFF

----------------

L20 → L27
SEASC OFF
NO INJECTION
```

No AkbasCore steering vector is injected into L20-L27.

This is not a reduced-dose region.

It is not another controller stage.

**The motor is off.**

---

# 11. Sessiz Kuyruk — Silent Tail

The post-intervention region:

```text
L20 → L27
```

is called:

**Sessiz Kuyruk — Silent Tail**

Inside the Sessiz Kuyruk:

```text
SEASC injection = 0
steering hooks   = absent
new Pusula dose  = 0
```

The ordinary transformer continues processing the state.

This creates a clean experimental question:

> **What does the transformer do with an upstream displacement after the mechanism that created it has stopped?**

---

# 12. İz Bırakma — Trace Imprinting

The L0-L19 intervention creates a measurable difference between the Vanilla and SEASC trajectories.

AkbasCore calls the creation of this downstream-visible difference:

**İz Bırakma — Trace Imprinting**

This does not mean that the originally injected vector survives unchanged.

It means that active intervention creates a measurable state difference that subsequent transformer computation can process.

---

# 13. İz Taşınımı — Trace Transport

After the motor stops:

\[
\Delta_L
=
h_L^{SEASC}
-
h_L^{Vanilla}
\]

for:

\[
L\ge20
\]

can continue changing.

AkbasCore calls this:

**İz Taşınımı — Trace Transport**

Relative displacement can be measured as:

\[
R_L
=
\frac{
\|h_L^{SEASC}-h_L^{Vanilla}\|_2
}{
\|h_L^{Vanilla}\|_2
}
\]

The downstream difference may:

- rotate;
- grow;
- contract;
- reorient;
- change projection.

Therefore:

\[
\text{Injected Vector}
\neq
\text{Downstream Net Displacement}
\]

---

# 14. Taşınım İzi — Transport Trace

The ordered sequence:

\[
\Delta_{20},
\Delta_{21},
\dots,
\Delta_{27}
\]

is called:

**Taşınım İzi — Transport Trace**

It records the measured trajectory of the intervention-created difference through the Sessiz Kuyruk.

Current experiments indicate that these trajectories can be structured, prompt-replicable under controlled conditions, and concept-dependent.

No single universal semantic transport channel has been established.

---

# KIZILELMA — Live Intervention X-Ray

### [OPEN KIZILELMA LIVE X-RAY](./AkbasCore_SEASC_KIZILELMA_Live_X_Ray)

KIZILELMA makes the motor visible.

It shows:

```text
WHAT CREATED THE CONTRAST

        ↓

WHAT SYNTHETIC COMPASSES
WERE BUILT

        ↓

WHAT WAS INJECTED

        ↓

WHERE THE MOTOR STOPPED

        ↓

WHAT HAPPENED AFTERWARD

        ↓

WHETHER THE OUTPUT CHANGED
```

KIZILELMA records:

- matched semantic carriers;
- 20 real layer-local Pusulas;
- 71,680 compass values;
- requested layer dose;
- measured injection magnitude;
- Vanilla hidden states;
- SEASC hidden states;
- 28-layer relative displacement;
- the L19/L20 cutoff;
- Vanilla generated text;
- SEASC generated text;
- a weight-change sentinel.

---

## Overview X-Ray

The Overview poster exposes:

```text
EXTRACT
↓
BUILD
↓
INJECT
↓
STOP
↓
OBSERVE
```

It includes:

- semantic compass;
- steering target;
- real vector fingerprint;
- active L0-L19 region;
- motor-OFF L20-L27 region;
- hidden-state displacement;
- Vanilla output;
- SEASC output.

---

## Technical X-Ray

The Technical poster exposes:

```text
20 × 3584
REAL COMPASS FINGERPRINT

+

MEASURED INJECTION MAGNITUDE

+

28-LAYER
VANILLA → SEASC DISPLACEMENT

+

LOCKED ENVELOPE METADATA
```

The Overview and Technical records separate public visual understanding from deeper technical inspection.

---

# Why KIZILELMA Matters

Generated text alone can hide internal intervention effects.

In a recent reference demonstration:

```text
Vanilla:
The bird is perched on the branch.

SEASC:
The bird is perched on the branch.
```

The greedy-decoded text remained identical.

The internal trajectories did not.

In that individual run, relative Vanilla→SEASC displacement was already substantial near the end of active steering and continued evolving after the motor stopped, reaching approximately the mid-40% range at L27.

This single run does not establish that every intervention behaves this way.

It demonstrates a methodological point:

> **No text change does not imply no internal intervention effect.**

The reverse is equally important:

> **A large hidden-state displacement does not automatically imply behavioral success.**

KIZILELMA therefore displays both internal measurements and generated output.

---

# Hidden-State X-Ray

This distinction became particularly clear in TEST 142.

For every layer:

\[
\Delta_L
=
h_L^{SEASC}
-
h_L^{Vanilla}
\]

Representative TEST 142 measurements:

```text
Mean relative displacement ≈ 30.37%
Mean cos(Δ,A)              ≈ 0.452

L00 relative displacement  ≈ 10%
L19 relative displacement  ≈ 36.90%
```

The important distinction is:

```text
u = intentional injection

Δ = downstream net difference

u ≠ Δ
```

The intentionally injected vector and the downstream net displacement are different mathematical objects.

---

# Downstream Transport Research

Once intervention and observation were separated, a new question became possible:

> **Can the motor-OFF evolution of an intervention trace be predicted from measured local geometry?**

TEST 192-197 investigated this question.

---

# 15. Yerel Taşınım İşleci

## Local Transport Operator

Low-rank mappings were estimated between neighboring downstream displacement states.

Conceptually:

\[
\widehat{\Delta}_{L+1}
=
T_L\Delta_L
\]

AkbasCore calls this experimental mapping:

**Yerel Taşınım İşleci — Local Transport Operator**

These operators are:

- local;
- empirical;
- low-rank;
- support-dependent.

They are **not** claimed to be the full transformer Jacobian.

They do not recover complete 3,584-dimensional transformer dynamics.

---

# 16. Taşınım Zinciri — Transport Chain

Local predictions can be recursively composed:

\[
\widehat{\Delta}_{L+2}
=
T_{L+1}T_L\Delta_L
\]

AkbasCore calls the resulting recursive sequence:

**Taşınım Zinciri — Transport Chain**

This creates two evaluation regimes:

```text
Teacher-Forced Prediction

versus

Recursive Free-Run Prediction
```

Recursive prediction is stricter because prediction errors can propagate through the chain.

---

# 17. İnşa Uzayı — BUILD Span

The controlled low-dimensional support used to estimate transport operators is called:

**İnşa Uzayı — BUILD Span**

For target displacement \(d\) and BUILD-support projector \(P_B\):

\[
\mathrm{Coverage}(d)
=
\frac{
\|P_Bd\|_2^2
}{
\|d\|_2^2
}
\]

The BUILD span defines the geometric support available to the low-rank predictor.

---

# 18. Kapsama — Span Coverage

**Kapsama — Span Coverage**

measures how much of a target displacement is represented by available BUILD support.

Kapsama is a geometric measurement.

It is not itself a causal quantity.

TEST 196 found a strong association between increases in BUILD-span coverage and increases in novel-surface prediction quality.

That association motivated TEST 197.

---

# 19. İnşa Kapasitesi — BUILD Capacity

**İnşa Kapasitesi — BUILD Capacity**

is the controlled amount of support used to construct the İnşa Uzayı.

TEST 197 increased nested BUILD support:

```text
N08
N16
N24
N32
```

while keeping novel SURFACE evaluation fixed.

| BUILD N | Coverage | Teacher | Free L27 | Free Error |
|---:|---:|---:|---:|---:|
| 8 | 0.684935 | 0.669637 | 0.664327 | 0.740958 |
| 16 | 0.708929 | 0.692085 | 0.667969 | 0.738735 |
| 24 | 0.722478 | 0.704334 | 0.673609 | 0.733351 |
| 32 | 0.735142 | 0.716287 | 0.678708 | 0.727619 |

Across pooled samples:

```text
Δcoverage → Δprediction

Pearson  = +0.869363
Spearman = +0.816145
N        = 320
```

Coverage and teacher-forced prediction improved monotonically as BUILD support increased.

Recursive terminal behavior was more heterogeneous.

Terminal N32−N8 results:

```text
A / ABOVE↔BELOW
+0.018554
CI [+0.011713,+0.026236]

B / BEFORE↔AFTER
-0.014292
CI [-0.029952,+0.000813]

C / INSIDE↔OUTSIDE
+0.005791
CI [-0.006047,+0.016525]

D / OPEN↔CLOSED
+0.010982
CI [+0.001611,+0.022047]

E / CAUSE↔EFFECT
+0.050867
CI [+0.042421,+0.060804]
```

Recorded decision:

```text
BUILD_SPAN_CAPACITY_SCALING_PARTIALLY_REPLICATED
```

The supported conclusion is narrower than universal capacity scaling:

> **Increasing BUILD support consistently increased geometric coverage and teacher-forced prediction quality, while terminal recursive improvement remained semantic-axis dependent.**

---

# Composite Compass Foundry Research

AkbasCore previously developed a broader compass-construction line.

This remains part of the project lineage.

---

# 20. Banka Pusulası — Bank-Mean Compass

For matched activation pair \(i\):

\[
d_{L,i}
=
\operatorname{normalize}
(h^+_{L,i}-h^-_{L,i})
\]

The pair directions are aggregated:

\[
B_L
=
\operatorname{normalize}
\left(
\sum_i d_{L,i}
\right)
\]

AkbasCore calls this:

**Banka Pusulası — Bank-Mean Compass**

---

# 21. Subspace-Fisher Reference

A reliability-weighted reference direction was constructed using within-class residual geometry and a regularized low-rank inverse action.

Conceptually:

\[
F_L
=
\operatorname{normalize}
\left[
(C_L+\lambda_LI)^{-1}
(\mu_L^+-\mu_L^-)
\right]
\]

Subspace-Fisher uses standard mathematical machinery.

AkbasCore does not claim Fisher/LDA geometry, covariance inversion or SVD as project inventions.

Its role in this research line is reference geometry.

---

# 22. Artık Yön — Residual Direction

The Banka Pusulası can be decomposed relative to the Fisher reference:

\[
c_L
=
\langle F_L,B_L\rangle
\]

and:

\[
R_L
=
\operatorname{normalize}
(B_L-c_LF_L)
\]

AkbasCore calls \(R_L\):

**Artık Yön — Residual Direction**

---

# 23. Bileşik Pusula — Composite Compass

The experimental compass family is:

\[
A_L(\beta)
=
\operatorname{normalize}
\left[
c_LF_L
+
\beta
\sqrt{1-c_L^2}R_L
\right]
\]

where \(\beta\) is:

**Bileşim Katsayısı — Composition Coefficient**

Under the reference orientation:

```text
β = 0 → Fisher reference
β = 1 → Bank-Mean
```

The AkbasCore 3.2 reference lock used:

```text
β = 1.0
```

Representative TEST 162 measurements:

| Scale | Fisher | Bank-Mean | Bank − Fisher |
|---:|---:|---:|---:|
| 0.25 | 0.090148 | 0.082974 | −0.007174 |
| 0.50 | 0.167929 | **0.235962** | **+0.068033** |
| 1.00 | 0.116622 | **0.264574** | **+0.147951** |

This did not establish universal Bank-Mean superiority.

It showed a dose-dependent functional difference under the tested conditions.

---

# Runtime-to-Weight Research

AkbasCore also investigated whether runtime geometry could be partially retained in parameters.

This is not part of the current inference-time SEASC motor.

---

# 24. Kristalleşme — Crystallization

**Kristalleşme — Crystallization**

is the AkbasCore term for experiments attempting to transfer part of a runtime steering trajectory into parameter updates and then evaluate with the runtime motor switched off.

TEST 144 used:

```text
LoRA rank : 8
alpha     : 16
targets   : q/k/v/o
trainable : ~0.0662%
```

Final held-out progress was approximately:

```text
+15.79%
```

under that experiment.

This demonstrated partial retention.

It did not establish complete conversion of runtime steering into weights.

---

# 25. Koruma Kalkanı — Retention Guard

Sequential crystallization exposed parameter-space interference.

**Koruma Kalkanı — Retention Guard**

is the AkbasCore experimental mechanism for projecting a proposed optimizer displacement away from measured retention-sensitive parameter directions.

For displacement \(d\) and guard direction \(g\):

\[
\alpha
=
\frac{d\cdot g}{g\cdot g}
\]

when the corresponding guard condition is active.

The important research conclusion was:

> **Activation-space independence does not imply parameter-space independence.**

Retention Guard belongs to the parameter-space research line.

It is not part of the current runtime SEASC motor.

---

# Canonical AkbasCore Terminology

| Turkish canonical term | English gloss |
|---|---|
| **Karşıtlık Ocağı** | Contrast Forge |
| **Sentetik Pusula Üretimi** | Synthetic Compass Synthesis |
| **Pusula** | Compass |
| **Katman Pusulası** | Layer-Local Compass |
| **Pusula Demeti** | Compass Bundle |
| **Pusula Dökümhanesi** | Compass Foundry |
| **Banka Pusulası** | Bank-Mean Compass |
| **Artık Yön** | Residual Direction |
| **Bileşik Pusula** | Composite Compass |
| **Bileşim Katsayısı** | Composition Coefficient |
| **Dondurulmuş Norm** | Frozen Norm |
| **Fiziksel Doz** | Physical Dose |
| **DRA Zarfı** | DRA Envelope |
| **İvme** | Steering Dose |
| **Sönüm** | Decay Rate |
| **Zirve** | Amplitude |
| **Taban** | Non-Zero Floor |
| **Norma Bağlı Pusula Enjeksiyonu** | Norm-Anchored Compass Injection |
| **Kesim Sınırı** | Intervention Cutoff |
| **Sessiz Kuyruk** | Silent Tail |
| **İz Bırakma** | Trace Imprinting |
| **İz Taşınımı** | Trace Transport |
| **Taşınım İzi** | Transport Trace |
| **Yerel Taşınım İşleci** | Local Transport Operator |
| **Taşınım Zinciri** | Transport Chain |
| **İnşa Uzayı** | BUILD Span |
| **Kapsama** | Span Coverage |
| **İnşa Kapasitesi** | BUILD Capacity |
| **Kristalleşme** | Crystallization |
| **Koruma Kalkanı** | Retention Guard |
| **KIZILELMA** | Live Intervention X-Ray |
| **DRA** | Damped Resonance Alignment |
| **SEASC** | Standardized Evidential Activation-Steering Core |

These names identify AkbasCore architectural components, measurements and experimental abstractions.

Standard mathematical operations retain their established scientific provenance.

---

# What Is Actually Running?

The current KIZILELMA runtime path is:

```text
Karşıtlık Ocağı
        ↓
Sentetik Pusula Üretimi
        ↓
20 Katman Pusulası
        ↓
Pusula Demeti
        ↓
DRA Zarfı
        ↓
Dondurulmuş Norm
        ↓
Fiziksel Doz
        ↓
Norma Bağlı Pusula Enjeksiyonu
        ↓
L0 → L19
        ↓
Kesim Sınırı
        ↓
Sessiz Kuyruk
L20 → L27
        ↓
İz Taşınımı
        ↓
KIZILELMA X-RAY
```

It does **not** silently run:

- Fisher controller;
- Bank controller;
- cosine gate;
- dynamic brake;
- saturation controller;
- runtime cap;
- lens controller;
- radar;
- JVP controller;
- transport operator;
- BUILD predictor;
- Retention Guard;
- LoRA;
- weight modification;
- dynamic feedback controller.

Those belong to research history or experimental measurement branches.

---

# Reference Configuration

```text
Model          : Qwen/Qwen2.5-7B-Instruct
Transformer    : 28 decoder layers
Hidden size    : 3584
Model dtype    : BF16
Compass math   : FP32
Steered layers : L0-L19
Silent Tail    : L20-L27
Runtime        : CUDA
Reference GPU  : NVIDIA A100
Decoding       : deterministic / greedy

İvme           : 0.10
Sönüm          : 0.30
Zirve          : 0.70
Taban          : 0.20

Base RSS       : ≈ 0.250235055
```

---

# Experimental Lineage

```text
DRA
↓
Closed-Loop Activation Steering
↓
Synthetic Concept Experiments
↓
SEASC
↓
Hidden-State X-Ray
↓
Kristalleşme
↓
Cross-Vector Interference
↓
Koruma Kalkanı
↓
Natural / Causal Transport
↓
Semantic Transport
↓
Fisher ↔ Bank Decomposition
↓
Motor-OFF Tail Experiments
↓
İz Taşınımı
↓
Yerel Taşınım İşleçleri
↓
Novel-Surface Transfer
↓
İnşa Uzayı
↓
Kapsama
↓
İnşa Kapasitesi
↓
TEST 197
↓
KIZILELMA LIVE X-RAY
```

Negative and falsifying results remain part of the record.

Mechanisms are not retained merely because they once existed.

---

# What the Evidence Currently Supports

### AkbasCore can synthesize layer-local steering directions from model activations

The current implementation constructs twenty independent 3,584-dimensional Pusulas from matched internal activation contrasts.

### Physical intervention magnitude is measurable

SEASC expresses requested displacement relative to the pre-intervention hidden-state norm.

### Direction, magnitude and depth are separated

Pusula determines direction.

SEASC determines physical intervention magnitude.

DRA determines the depth profile.

### Injection and downstream displacement are different objects

Later transformer computation can rotate and alter an upstream perturbation.

### Active intervention can be stopped

L20-L27 can be observed without new SEASC injection.

### Motor-OFF traces contain measurable structure

Controlled experiments found prompt- and concept-dependent downstream trajectories.

### Low-rank local transport prediction is possible under controlled support

BUILD-span operators predict part of the downstream trace.

### Some prediction transfers to novel surface formulations

Prediction is not entirely confined to exact BUILD wording.

### BUILD-span coverage is strongly associated with prediction quality

TEST 196-197 measured a strong relationship under the tested setup.

This association is not treated as causal proof.

### More support does not universally solve recursive prediction

TEST 197 was deliberately recorded as partially replicated.

### Internal change and output change are different measurements

A substantial hidden-state displacement can coexist with an unchanged greedy output.

---

# What Is Not Claimed

AkbasCore does **not** currently establish:

- that contrastive activation subtraction itself was invented by AkbasCore;
- that difference-of-means is a new mathematical operation;
- a universal semantic representation;
- a universal transformer coordinate system;
- a universal full-rank transport operator;
- recovery of the complete transformer Jacobian;
- complete prediction of transformer dynamics;
- universal superiority of one compass-construction method;
- universal monotonic benefit from BUILD capacity;
- formal causal identification from the CAUSE↔EFFECT semantic axis;
- prompt-independent semantic control;
- cross-model generalization;
- cross-language generalization;
- a universal AI alignment solution;
- safety guarantees;
- human-level hidden-state interpretability.

The strongest current evidence is concentrated on controlled experiments with Qwen2.5-7B-Instruct.

Broader replication remains open.

---

# Reproducibility Philosophy

AkbasCore follows one central experimental rule:

> **Measure the mechanism separately from the generated text.**

The research therefore distinguishes:

```text
synthetic compass
requested dose
realized dose
injected displacement
downstream displacement
transport trace
transport prediction
span coverage
behavioral output
collateral change
weight change
```

These quantities are not treated as interchangeable.

A convincing generated answer is not sufficient evidence.

A large hidden-state displacement is not sufficient evidence of behavioral success.

A correlation is not automatically treated as causation.

A low-rank predictor is not automatically called the full system dynamics.

Negative and falsifying results are preserved.

---

# Project Status

AkbasCore is currently centered on a deliberately measurable motor:

```text
MODEL SYNTHESIZES
LAYER-LOCAL COMPASSES

        ↓

SEASC MEASURES
PHYSICAL INTERVENTION

        ↓

DRA SHAPES
INTERVENTION THROUGH DEPTH

        ↓

CUDA APPLIES
THE INTERVENTION

        ↓

THE MOTOR STOPS

        ↓

THE TRANSFORMER CONTINUES

        ↓

KIZILELMA OBSERVES
WHAT HAPPENS NEXT
```

The immediate research direction is not to add another runtime controller.

Open questions include:

- synthetic compass stability;
- semantic-axis generalization;
- downstream transport;
- support geometry;
- transport capacity;
- novel-surface transfer;
- model scale;
- model family;
- language;
- dose regime.

---

# Current AkbasCore Research Tree

```text
AKBASCORE

PUSULA DÖKÜMHANESİ
- Karşıtlık Ocağı
- Sentetik Pusula Üretimi
- Katman Pusulası
- Pusula Demeti
- Banka Pusulası [research]
- Subspace-Fisher [reference]
- Artık Yön [research]
- Bileşik Pusula [research]

SEASC
- Dondurulmuş Norm
- Fiziksel Doz
- Norma Bağlı Pusula Enjeksiyonu

DRA ZARFI
- İvme
- Sönüm
- Zirve
- Taban

MÜDAHALE SINIRI
- Kesim Sınırı
- Sessiz Kuyruk

TAŞINIM ARAŞTIRMASI
- İz Bırakma
- İz Taşınımı
- Taşınım İzi
- Yerel Taşınım İşleci
- Taşınım Zinciri
- İnşa Uzayı
- Kapsama
- İnşa Kapasitesi

PARAMETRE-UZAYI ARAŞTIRMASI
- Kristalleşme
- Koruma Kalkanı

KIZILELMA
- Overview X-Ray
- Technical X-Ray
```

> **The model provides the geometry.  
> The Foundry synthesizes the compasses.  
> SEASC measures the dose.  
> DRA shapes the depth.  
> CUDA applies the intervention.  
> The motor stops.  
> KIZILELMA watches the trace.  
> The weights remain frozen.**

---

# Citation

For scientific or technical work based materially on AkbasCore, please cite the relevant **versioned AkbasCore release and associated archival record** rather than treating the continuously updated `main` branch as a fixed publication.

The repository's timestamped commits, Releases, experiment records and archival deposits preserve the development lineage.

**Project:** AkbasCore  
**Developer / Researcher:** Mustafa Akbaş  
**Research lineage:** DRA → SEASC → Transport → KIZILELMA  
**Year:** 2026  

---

# License

**AkbasCore software is currently distributed under the license terms provided in this repository.**

**Copyright © 2026 Mustafa Akbaş — AkbasCore**

The repository contains software together with experimental methodology, research results, architectural terminology and attribution information.

For the complete and controlling terms, scope, copyright notice and attribution information, read the repository's [`LICENSE`](LICENSE) file.

Standard mathematical techniques used within AkbasCore retain their existing scientific provenance.

---

# AkbasCore — SEASC

**Mustafa Akbaş · 2026**

> **Direction is synthesized.  
> Dose is measured.  
> Depth is shaped.  
> The motor stops.  
> The trace remains visible.**

## SEE THE INTERVENTION, NOT JUST THE OUTPUT.

### [KIZILELMA LIVE X-RAY](./AkbasCore_SEASC_KIZILELMA_Live_X_Ray)
