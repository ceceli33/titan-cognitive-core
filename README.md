AkbasCore 3.2 — SEASC + Composite Compass Foundry


Standardized Evidential Activation-Steering Core




An inference-time, layerwise activation-steering motor with measured physical dose and a model-derived composite compass.




Author: Mustafa Akbaş

Status: Independent research / experimental reference implementation

Current version: AkbasCore 3.2 — September 2026

Reference model: Qwen/Qwen2.5-7B-Instruct

Reference hardware: NVIDIA A100

Weights modified during steering: No

Fine-tuning required: No




AkbasCore 3.2 is an experimental Reference Core, not a claim of a universal production alignment system.





What Is AkbasCore?


AkbasCore 3.2 is a single integrated inference-time activation-steering motor and experimental methodology.


It intervenes directly in a transformer's hidden states during the forward pass without modifying the model's weights.


Internally, the motor solves three connected problems:






Where should the hidden state be pushed?

→ Pusula Dökümhanesi / Composite Compass Foundry






How much physical displacement should be applied?

→ SEASC frozen-norm direct dose






How should that dose vary with transformer depth?

→ Static normalized DRA envelope






These are functional components of one AkbasCore motor, not separate systems.


Matched activation examples
          │
          ▼
Pusula Dökümhanesi
Composite Compass Foundry
          │
          ▼
Bileşik Pusula A_L
Layerwise steering direction
          │
          ▼
SEASC
Frozen-norm physical dose
          │
          ▼
DRA Zarfı
Static depth envelope
          │
          ▼
Hidden-state intervention
          │
          ▼
Transformer continues



No model weights are changed during steering.


The 3.2 Reference Core contains no runtime classifier, learned controller, cosine-magnitude gate, dynamic brake, saturation controller, cap, JVP, radar, or dynamic feedback controller.



Why 3.2 Exists


AkbasCore began with a simple question:




Can the internal trajectory of a transformer be influenced directly during inference, with a mathematically controlled intervention, rather than by retraining the model or rewriting its prompt?




The project evolved through several generations of experiments.


Early versions explored embedding-derived compasses and closed-loop cosine control. Later experiments on Qwen2.5-7B-Instruct showed that several of those mechanisms either became ineffective, were unnecessary, or did not justify their complexity.


The architecture was therefore progressively reduced.


The current motor is deliberately simpler:


Direction  → derived from model activations
Magnitude  → explicit fraction of pre-intervention hidden-state norm
Depth      → static normalized DRA envelope
Injection  → direct frozen-norm CUDA operation



AkbasCore 3.2 continues this simplification.


The SEASC physical-dose motor remains unchanged from the 3.0/3.1 line.


3.2 changes the compass construction, not the SEASC injection law.



AkbasCore 3.2 Architecture


1. Pusula — Compass


Pusula is the canonical AkbasCore term for the unit direction along which the motor applies its intervention.


For layer L:


[
A_L \in \mathbb{R}^{H},
\qquad
|A_L|_2 = 1
]


AkbasCore constructs a separate Pusula for every steered layer.


The current reference configuration steers:


L0 → L19



of the 28-layer Qwen2.5-7B-Instruct model.


There is no single global steering vector broadcast unchanged across all twenty layers.



2. Pusula Dökümhanesi — Compass Foundry


Pusula Dökümhanesi (Compass Foundry) is the direction-construction mechanism inside the AkbasCore motor.


It constructs the layerwise Pusula before generation begins.


Its responsibility is direction, not magnitude:


Pusula Dökümhanesi → direction
SEASC               → physical displacement
DRA Zarfı           → layerwise dose profile



This separation is fundamental to the current architecture.


The Foundry determines the direction of intervention.


SEASC determines the physical magnitude of intervention.


The DRA Zarfı determines how that magnitude changes with depth.


Together they form the AkbasCore motor.



Composite Compass Foundry


AkbasCore 3.2 retains two related pieces of directional information:




Subspace-Fisher Reference


Banka Pusulası / Bank-Mean Compass




The relationship between them is explicitly measured and decomposed.



3. Subspace-Fisher Reference


A reliability-weighted reference direction is computed from matched positive and negative activation samples using within-class residual geometry and a regularized low-rank inverse action.


Conceptually:


[
F_L =
\operatorname{normalize}
\left[
(C_L+\lambda_L I)^{-1}
(\mu_L^+-\mu_L^-)
\right]
]


The implementation does not construct or invert a full dense H \times H covariance matrix.


Reduced SVD operations are performed in FP32.


Subspace-Fisher is standard mathematical machinery and is not claimed as an original mathematical invention of AkbasCore.


Within AkbasCore 3.2, its role is that of a reference geometry.



4. Banka Pusulası — Bank-Mean Compass


Banka Pusulası (Bank-Mean Compass) is constructed from a bank of matched positive/negative activation-pair directions.


For matched pair i:


[
d_{L,i}


\operatorname{normalize}
\left(
h^+{L,i}-h^-{L,i}
\right)
]


The pair directions are consistently oriented relative to the reference hemisphere and then aggregated:


[
B_L


\operatorname{normalize}
\left(
\sum_i d_{L,i}
\right)
]


In the current AkbasCore 3.2 reference lock, B_L is the runtime steering direction.



5. Artık Yön — Residual Direction


The Banka Pusulası is decomposed relative to the Subspace-Fisher reference.


First:


[
c_L=\langle F_L,B_L\rangle
]


Then the component of B_L orthogonal to F_L is isolated:


[
R_L


\operatorname{normalize}
\left(
B_L-c_LF_L
\right)
]


AkbasCore calls R_L:


Artık Yön — Residual Direction


For unit F_L and B_L, the magnitude of the unnormalized orthogonal component is:


[
|B_L-c_LF_L|_2


\sqrt{1-c_L^2}
]


This decomposition allows the Fisher reference and Bank-Mean actuator to be represented inside one compass family.



6. Bileşik Pusula — Composite Compass


The AkbasCore 3.2 compass family is:


[
A_L(\beta)


\operatorname{normalize}
\left[
c_LF_L
+
\beta
\sqrt{1-c_L^2}
R_L
\right]
]


where β is the:


Bileşim Katsayısı — Composition Coefficient


Pair directions are oriented so that the Bank-Mean direction lies in the positive Fisher reference hemisphere, giving c_L \ge 0.


Under that orientation:


β = 0  → Subspace-Fisher reference direction
β = 1  → Bank-Mean direction



At \beta=1:


[
c_LF_L+\sqrt{1-c_L^2}R_L=B_L
]


up to numerical precision.


The current AkbasCore 3.2 reference lock is:


β = 1.0



Therefore:


Subspace-Fisher remains the reference geometry.

Banka Pusulası is the current runtime actuator.


The 3.2 architecture does not discard the Fisher geometry; it explicitly measures how the runtime Bank-Mean direction differs from it.



Why Bank-Mean?


The Bank-Mean choice was not made from a single attractive generated output.


The experimental line progressively separated:




natural transport geometry;


causal transport;


semantic transport;


context dependence;


state routing;


semantic direction-family effects;


Fisher geometry;


Bank-specific residual geometry.




The decisive Fisher → Bank-Mean decomposition showed a dose-dependent difference.


Representative measurements from TEST 162:




Scale
Fisher
Bank-Mean
Bank − Fisher




0.25
0.090148
0.082974
−0.007174


0.50
0.167929
0.235962
+0.068033


1.00
0.116622
0.264574
+0.147951




Paired bootstrap results for the Bank − Fisher difference were approximately:


Scale 0.25 → not significant
Scale 0.50 → p ≈ 0.0188
Scale 1.00 → p ≈ 0.00075



This does not establish that Bank-Mean is universally superior to Fisher.


At the lowest tested scale, Fisher was slightly stronger.


The supported conclusion is narrower:




Fisher and Bank-Mean share substantial geometry, while the Bank-specific residual contributes additional causal steering effect in the tested moderate/high-dose regime.




That result motivated the current \beta=1.0 reference lock.



SEASC Motor


SEASC — Standardized Evidential Activation-Steering Core is the physical injection mechanism inside AkbasCore.


For hidden state h_L, unit compass A_L, and relative layer dose \rho_L:


[
h'_L


h_L
+
\rho_L
|h_L|_{\mathrm{pre}}
A_L
]


with:


[
\rho_L


\mathrm{İvme}\cdot E(L)
]


The hidden-state norm is measured before the intervention.


This gives the requested intervention an explicit physical interpretation relative to the state being modified.



7. Dondurulmuş Norm — Frozen Norm


AkbasCore calls the pre-intervention hidden-state norm:


Dondurulmuş Norm — Frozen Norm


[
n_L


|h_L|_{\mathrm{pre}}
]


It is captured before the injection modifies the hidden vector.


The requested displacement is then:


[
\Delta h_L


\rho_L n_L A_L
]


The intervention therefore cannot increase its own requested dose by first increasing the norm from which that dose is calculated.



8. Fiziksel Doz — Physical Dose


Fiziksel Doz is the canonical AkbasCore term for displacement relative to the pre-intervention hidden-state norm:


[
D_L


\frac{|\Delta h_L|2}
{|h_L|{\mathrm{pre}}}
]


Because A_L is a unit vector:


[
D_L \approx \rho_L
]


up to implementation and numerical precision.


If:


İvme = 0.10
E(0) = 1



then the requested first-layer displacement is:


10% of the pre-intervention hidden-state norm



rather than an arbitrary coefficient whose realized magnitude depends on instantaneous hidden-state alignment.



DRA Zarfı — DRA Envelope


AkbasCore retains the critically damped depth profile introduced in the earlier DRA line.


The base profile is:


[
kb(L)


\mathrm{Zirve},
e^{-\mathrm{Sönüm}L}
(1+\mathrm{Sönüm}L)
+
\mathrm{Taban}
]


It is normalized at the first steered layer:


[
E(L)


\frac{kb(L)}{kb(0)}
]


therefore:


[
E(0)=1
]


and:


[
\rho_L


\mathrm{İvme},E(L)
]


The current reference constants are:




Canonical term
English gloss
Value




İvme
Initial Physical Steering Dose
0.10


Sönüm
Decay Rate
0.30


Zirve
Envelope Amplitude
0.70


Taban
Non-Zero Floor
0.20




The resulting reference dose schedule is:


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



The DRA Zarfı controls the depth profile of the physical intervention.


It does not determine the semantic direction.



Canonical AkbasCore 3.2 Terminology


The following Turkish terms form the canonical terminology used by the AkbasCore project for the current motor architecture.




Turkish canonical term
English gloss
Meaning in AkbasCore




Pusula
Compass
Layerwise unit steering direction


Pusula Dökümhanesi
Compass Foundry
Direction-construction mechanism


Banka Pusulası
Bank-Mean Compass
Central direction aggregated from matched activation-pair contrasts


Artık Yön
Residual Direction
Bank-Mean component orthogonal to the Fisher reference


Bileşik Pusula
Composite Compass
β-controlled Fisher/Bank compass construction


Bileşim Katsayısı (β)
Composition Coefficient
Controls movement from Fisher reference toward Bank-Mean


Dondurulmuş Norm
Frozen Norm
Hidden-state norm captured before intervention


Fiziksel Doz
Physical Dose
Displacement relative to the frozen hidden-state norm


Zarf
Envelope
Normalized depth-dependent DRA dose profile


İvme
Steering Dose
First-layer physical dose control


Sönüm
Decay Rate
Controls DRA envelope decay


Zirve
Amplitude
DRA envelope amplitude


Taban
Permanent Floor
Non-zero residual component of the DRA envelope


DRA
Damped Resonance Alignment
Depth-dependent dose-envelope lineage


SEASC
Standardized Evidential Activation-Steering Core
Frozen-norm direct-dose injection mechanism




These terms identify components and measurements inside the documented AkbasCore motor and methodology.


Standard mathematical tools used within the implementation — including SVD, covariance estimation, regularized inverse operations, Fisher/LDA-style geometry, cosine similarity, difference-of-means and vector normalization — retain their standard mathematical meanings and provenance and are not claimed as AkbasCore inventions.



What Was Removed


The current architecture is intentionally smaller than several historical AkbasCore branches.


The 3.2 Reference Core contains:


Composite Compass Foundry
        +
Frozen-Norm Direct Dose
        +
Static Normalized DRA Envelope



It does not contain:


cosine magnitude gate
dynamic brake
saturation controller
runtime cap
lens measurement path
multi-axis manifold
radar
JVP transport controller
dynamic runtime controller
LoRA
runtime weight modification



Some of these mechanisms produced useful experimental findings.


Others were rejected.


They are not active components of the current runtime motor.



Experimental Lineage


The repository preserves the development history rather than rewriting earlier experiments after later findings.


In broad terms:


Early DRA
   │
   ▼
Live hidden-state intervention
   │
   ▼
Synthetic concept injection
   │
   ▼
SEASC standardized physical dose
   │
   ▼
Hidden-state X-ray
   │
   ▼
Runtime → weight crystallization
   │
   ▼
Cross-vector interference
   │
   ▼
Retention Guard
   │
   ▼
Natural fluctuation / transport atlas
   │
   ▼
Causal potent/null transport
   │
   ▼
Relay and re-encoding experiments
   │
   ▼
Semantic transport
   │
   ▼
Context and semantic-object experiments
   │
   ▼
State-routing tests
   │
   ▼
Semantic direction-family assay
   │
   ▼
Fisher → Bank-Mean causal decomposition
   │
   ▼
AkbasCore 3.2



Negative and falsifying results are intentionally retained.


Mechanisms that failed to earn their complexity were not accumulated into the final motor.


Detailed historical experiments, equations, code and logs remain available through versioned repository records, Releases and archival records.



Reference Configuration


Model          : Qwen/Qwen2.5-7B-Instruct
Transformer    : 28 decoder layers
Hidden size    : 3584
Model dtype    : BF16
Foundry math   : FP32
Steered layers : L0-L19
Runtime        : CUDA
Reference GPU  : NVIDIA A100
Decoding       : deterministic / greedy

İvme           : 0.10
Sönüm          : 0.30
Zirve          : 0.70
Taban          : 0.20
β              : 1.0



The reference CUDA kernel performs frozen-norm direct injection.


Conceptually:


1. Measure pre-intervention hidden-state norm
2. Compute requested physical dose
3. Apply the layerwise unit Pusula
4. Continue the transformer forward pass




AkbasCore 3.2 Reference Run


Reference environment:


Model      : Qwen/Qwen2.5-7B-Instruct
GPU        : NVIDIA A100-SXM4-80GB
Model dtype: BF16
Foundry    : FP32
β          : 1.0



Compass telemetry:


Mean cos(Fisher, Bank-Mean) : +0.938318
Mean Bank residual norm     :  0.333972
Mean cos(Composite, Bank)   :  0.999999997



With:


β = 1.0



the Composite Compass reconstructs the Bank-Mean actuator to numerical precision in this reference run.


Physical-dose telemetry:


Mean requested dose : 4.9801%
Mean realized dose  : 4.9833%

L0 requested        : 10.0000%
L0 realized         : 10.0019%

L19 requested       : 2.3966%
L19 realized        : 2.4000%



At L0, the absolute requested-versus-realized difference was approximately:


0.0019 percentage points



These measurements verify operation of the reference compass and physical-dose implementation.


They do not, by themselves, establish behavioral efficacy.



What the Evidence Currently Supports


The accumulated experiments support several narrower conclusions.


Physical dose is controllable


The SEASC CUDA implementation closely realizes the requested displacement under the tested BF16 configuration.


Injection and downstream displacement are different objects


A relatively small intervention applied upstream can rotate and change magnitude as subsequent transformer blocks process it.


Natural cross-layer activity contains strong low-rank structure


Controlled natural-fluctuation experiments found substantial low-dimensional cross-layer covariance.


Natural covariance does not imply direct causal transport


Causal intervention experiments separated observational transport structure from direct steering privilege.


Semantic geometry changes with depth


A single fixed semantic coordinate does not adequately describe the observed cross-layer behavior.


Context-dependent structure is measurable


Matched contexts produce measurable geometric structure, but context matching has not been established as a complete causal steering address.


Activation-space independence does not imply parameter-space independence


Sequential crystallization experiments showed substantial parameter-update interference despite low activation-space cosine between the tested axes.


Fisher and Bank-Mean are related but functionally non-identical


The Bank-specific residual can contribute additional causal steering effect in the tested moderate/high-dose regime.


Steering geometry is dose-dependent


The relative behavior of Fisher and Bank-Mean changes across the tested physical-dose regime.


These observations motivated the current Composite Compass Foundry.



What Is Not Claimed


AkbasCore 3.2 does not currently establish:




a universal representation geometry for transformer models;


universal superiority of Bank-Mean over Fisher;


a universal semantic subspace;


a general-purpose AI alignment solution;


semantic control independent of prompt or context;


cross-model generalization;


cross-language generalization;


safety guarantees;


human-level interpretability of hidden states;


that every measured low-rank structure is causal;


that standard mathematical components used inside the motor were invented by this project.




The current reference evidence is concentrated on controlled experiments using Qwen2.5-7B-Instruct.


Cross-model, cross-language and broader semantic validation remain open.



Reproducibility Philosophy


AkbasCore experiments follow a simple rule:




Measure the mechanism separately from the generated text.




Accordingly, experiments may record:




requested physical dose;


realized physical dose;


hidden-state displacement;


compass geometry;


downstream transport;


behavioral effect;


collateral language-model change;


random controls;


shuffled controls;


null distributions;


untouched validation/final splits where appropriate.




A visually convincing generated response is not treated as sufficient evidence by itself.


Negative and falsifying experiments are part of the research record.



Historical Versions


Older AkbasCore releases remain part of the project lineage.


They should be interpreted according to the architecture documented in their own release rather than retroactively as AkbasCore 3.2 systems.


Historical mechanisms including:


Sensör
Mercek
Fren
Doyum
Karşıt
Sınır
Manifold
Radar / JVP



belong to earlier experimental branches and are not active components of AkbasCore 3.2.


Likewise, runtime-to-weight crystallization and Retention Guard experiments belong to the experimental research line rather than the current inference-time motor.


The repository history, versioned Releases and archival records remain the appropriate sources for those stages.



Project Status


AkbasCore 3.2 is an experimental Reference Core.


The current motor is intentionally frozen around:


Composite Compass Foundry
          +
SEASC Frozen-Norm Direct Dose
          +
Static Normalized DRA Envelope



The immediate research objective is not to add another runtime controller.


It is to determine how well the current motor generalizes across:


semantic axes
contexts
model scales
model families
languages
dose regimes



while preserving the simplicity and measurability of the Reference Core.



Current Core


AKBASCORE 3.2
│
├── Pusula Dökümhanesi
│   │
│   ├── Subspace-Fisher Reference
│   │
│   ├── Banka Pusulası
│   │
│   ├── Artık Yön
│   │
│   └── Bileşik Pusula
│       └── β = 1.0
│
├── SEASC
│   │
│   ├── Dondurulmuş Norm
│   │
│   └── Fiziksel Doz
│
└── DRA Zarfı
    │
    ├── İvme
    ├── Sönüm
    ├── Zirve
    └── Taban





Direction is constructed. Dose is measured. Depth is shaped. Weights remain frozen.





Citation


For scientific or technical work based materially on AkbasCore, please cite the relevant versioned AkbasCore release and associated archival record rather than treating the continuously updated main branch as a fixed publication.


The repository's timestamped commits, Releases, experiment records and archival deposits preserve the development lineage of DRA, SEASC and the current AkbasCore architecture.


Project: AkbasCore

Developer / Researcher: Mustafa Akbaş

Current version: AkbasCore 3.2 — SEASC + Composite Compass Foundry

Year: 2026



License


AkbasCore software is released under the MIT License.


Copyright © 2026 Mustafa Akbaş — AkbasCore


The MIT License permits broad use, modification and redistribution of the software subject to its terms.


The AkbasCore research record also contains scientific methodology, experimental results, architectural terminology and attribution information that should not be inferred solely from this short README license summary.


For the complete and controlling software-license terms, copyright notice, scope information and project attribution information, read the repository's LICENSE file.


Standard mathematical techniques used within AkbasCore retain their existing scientific provenance.



AkbasCore 3.2 — SEASC + Composite Compass Foundry

Mustafa Akbaş · 2026


