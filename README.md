AkbasCore — SEASC


Standardized Evidential Activation-Steering Core




A model-internal steering motor that synthesizes its own layer-local compasses, applies measured physical intervention, stops deliberately, and observes what the transformer does next.




🔴 KIZILELMA LIVE X-RAY


SEE THE INTERVENTION, NOT JUST THE OUTPUT.


▶ OPEN KIZILELMA LIVE X-RAY


SYNTHESIZE → INJECT → SHAPE → STOP → OBSERVE



Author: Mustafa Akbaş

Status: Independent research / experimental reference implementation

Current research line: AkbasCore SEASC — September 2026

Reference model: Qwen/Qwen2.5-7B-Instruct

Reference hardware: NVIDIA A100

Model weights modified during steering: No

Fine-tuning required: No

LoRA required: No

External steering vector required: No




AkbasCore is an experimental Reference Core and research methodology, not a claim of a universal production alignment system.





The Core Idea


AkbasCore does not begin with a stored steering vector.


It does not load a behavioral adapter.


It does not fine-tune the model.


It does not modify the model weights.


Instead, the current AkbasCore motor uses the transformer's own internal activation geometry to synthesize the directions that will subsequently be used to intervene in that same model.


For the reference Qwen2.5-7B-Instruct configuration:


20 independently synthesized layer-local compasses
×
3,584 dimensions each
=
71,680 directional values



are constructed for the active intervention region.


The motor then combines these directions with:


model-internal compass synthesis
        +
frozen-norm physical dose
        +
critically damped depth shaping
        +
direct CUDA hidden-state injection
        +
explicit intervention cutoff
        +
motor-OFF downstream observation



This complete mechanism is the current AkbasCore SEASC architecture.



The AkbasCore Motor


MATCHED SEMANTIC CONTRASTS
          │
          ▼
   KARŞITLIK OCAĞI
    Contrast Forge
          │
          ▼
MODEL-INTERNAL ACTIVATIONS
          │
          ▼
SENTETİK PUSULA ÜRETİMİ
Synthetic Compass Synthesis
          │
          ├── L00 → A₀  ∈ R³⁵⁸⁴
          ├── L01 → A₁  ∈ R³⁵⁸⁴
          ├── L02 → A₂  ∈ R³⁵⁸⁴
          │
          │        ...
          │
          └── L19 → A₁₉ ∈ R³⁵⁸⁴
                     │
                     ▼
               PUSULA DEMETİ
               Compass Bundle
                20 × 3584
                     │
                     ▼
              DRA DOZ ZARFI
                     │
                     ▼
       DONDURULMUŞ NORM ÖLÇÜMÜ
                     │
                     ▼
     NORMA BAĞLI PUSULA ENJEKSİYONU
                     │
                     ▼
             L0 ───────── L19
                  SEASC ON
                     │
                     ▼
                KESİM SINIRI
                     │
                  MOTOR OFF
                     │
                     ▼
             L20 ───────── L27
                SESSİZ KUYRUK
                     │
                     ▼
                İZ TAŞINIMI
                     │
                     ▼
                 KIZILELMA
              LIVE X-RAY



The motor therefore separates six questions:




What semantic contrast are we measuring?


What direction does that contrast have inside each layer?


How large should the physical intervention be?


How should that dose vary through depth?


Where should active intervention stop?


What happens after the motor stops?





1. Karşıtlık Ocağı — Contrast Forge


Karşıtlık Ocağı is the AkbasCore stage that converts matched semantic oppositions into measurable internal activation contrasts.


Examples include:


ABOVE ↔ BELOW
BEFORE ↔ AFTER
INSIDE ↔ OUTSIDE
OPEN ↔ CLOSED
CAUSE ↔ EFFECT



Matched positive and negative carrier sentences are passed through the same model.


For every target layer L, AkbasCore records the corresponding hidden states:


$$
h_L^{POS}
\qquad\text{and}\qquad
h_L^{NEG}
$$


The mean activation of each side is then computed:


$$
\mu_L^{POS}


\operatorname{mean}(h_L^{POS})
$$


$$
\mu_L^{NEG}


\operatorname{mean}(h_L^{NEG})
$$


The purpose of Karşıtlık Ocağı is not to steer the model yet.


Its purpose is to expose a measurable contrast inside the model's own representation space.



2. Sentetik Pusula Üretimi — Synthetic Compass Synthesis


This is one of the central mechanisms of the current AkbasCore motor.


AkbasCore does not require a stored external steering vector.


Instead, the model participates in constructing the intervention directions from its own hidden-state geometry before steering begins.


For each layer:


$$
v_L^{sentetik}


\mu_L^{POS}


\mu_L^{NEG}
$$


The synthetic contrast is normalized:


$$
A_L


\frac{
v_L^{sentetik}
}{
|v_L^{sentetik}|_2
}
$$


producing a unit layer-local Pusula.


This process is repeated independently across the twenty active layers:


$$
A_0,A_1,\ldots,A_{19}
$$


For Qwen2.5-7B-Instruct:


$$
A_L\in\mathbb{R}^{3584}
$$


so one synthesis cycle produces:


20 independent layer-local vectors
×
3,584 real dimensions
=
71,680 directional values



before active steering begins.


This is fundamentally different from assuming that one fixed semantic vector should be copied unchanged through transformer depth.



3. Katman Pusulası — Layer-Local Compass


Each independently synthesized unit vector is called:


Katman Pusulası — Layer-Local Compass


For layer L:


$$
A_L\in\mathbb{R}^{H},
\qquad
|A_L|_2=1
$$


The current motor does not assume:


$$
A_0=A_1=\cdots=A_{19}
$$


Instead:


L00 has A₀
L01 has A₁
L02 has A₂
...
L19 has A₁₉



because semantic geometry is allowed to change through transformer depth.



4. Pusula Demeti — Compass Bundle


The complete collection:


$$
\mathcal{A}


{A_0,A_1,\ldots,A_{19}}
$$


is called:


Pusula Demeti — Compass Bundle


It represents the full layer-local directional structure used by one AkbasCore intervention.


For the reference model:


Pusula Demeti
=
20 × 3584
=
71,680 directional values



The Pusula Demeti is synthesized from the model itself.


It is not imported from an external vector library.



Important Scientific Context


Contrastive activation differences and steering-vector extraction are established ideas in activation-steering research.


AkbasCore therefore does not claim that subtracting positive and negative activations, by itself, is a new mathematical operation.


The AkbasCore contribution being investigated is the larger integrated architecture:


layer-local model-internal compass synthesis
        +
measured frozen-norm physical dose
        +
depth-dependent DRA shaping
        +
multi-layer direct intervention
        +
explicit motor cutoff
        +
motor-OFF downstream X-Ray
        +
transport measurement



and the experimental methodology built around that architecture.



5. Pusula Dökümhanesi — Compass Foundry


Pusula Dökümhanesi is the umbrella term for AkbasCore direction-construction mechanisms.


The current direct synthesis path is:


Pusula Dökümhanesi
        │
        ▼
Karşıtlık Ocağı
        │
        ▼
Sentetik Pusula Üretimi
        │
        ▼
Katman Pusulaları
        │
        ▼
Pusula Demeti



Historical Foundry experiments also include:


Banka Pusulası
Subspace-Fisher Reference
Artık Yön
Bileşik Pusula



These remain part of the experimental lineage but are not required for the direct KIZILELMA synthesis path.



6. Dondurulmuş Norm — Frozen Norm


Once the direction has been synthesized, AkbasCore must determine how strongly to intervene.


Before injection, the motor measures:


$$
n_L


|h_L|_{\mathrm{pre}}
$$


AkbasCore calls this:


Dondurulmuş Norm — Frozen Norm


The measurement occurs before the intervention modifies the state.


This is important because the intervention cannot increase the reference norm from which its own requested magnitude is calculated.



7. Fiziksel Doz — Physical Dose


AkbasCore defines intervention magnitude relative to the state being modified.


For relative layer dose \rho_L:


$$
\Delta h_L


\rho_L
|h_L|_{\mathrm{pre}}
A_L
$$


The relative physical displacement is:


$$
D_L


\frac{
|\Delta h_L|2
}{
|h_L|{\mathrm{pre}}
}
$$


Because:


$$
|A_L|_2=1
$$


the requested relative displacement satisfies:


$$
D_L\approx\rho_L
$$


up to implementation precision.


A requested 10% intervention therefore means approximately:




a displacement equal to 10% of the pre-intervention hidden-state norm




rather than an arbitrary uncalibrated coefficient.



8. DRA Zarfı — DRA Envelope


Physical dose is not constant through depth.


AkbasCore retains the critically damped depth profile developed in the DRA research line.


The base profile is:


$$
kb(L)


\mathrm{Zirve}
e^{-\mathrm{Sönüm}L}
(1+\mathrm{Sönüm}L)
+
\mathrm{Taban}
$$


normalized as:


$$
E(L)


\frac{kb(L)}{kb(0)}
$$


therefore:


$$
E(0)=1
$$


and:


$$
\rho_L


\mathrm{İvme},E(L)
$$


Current locked constants:




Turkish term
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




Reference schedule:


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



The locked base envelope has:


RSS(ρ) ≈ 0.250235055



The DRA Zarfı controls depth.


It does not determine semantic direction.



9. Norma Bağlı Pusula Enjeksiyonu — Norm-Anchored Compass Injection


The point where direction, state magnitude and depth dose meet is:


Norma Bağlı Pusula Enjeksiyonu — Norm-Anchored Compass Injection


For each active layer:


$$
\boxed{
h'_L


h_L
+
\rho_L
|h_L|_{\mathrm{pre}}
A_L
}
$$


This combines:


Katman Pusulası
       ×
Dondurulmuş Norm
       ×
Fiziksel Doz
       ×
DRA Zarfı



into one direct hidden-state intervention.


The current reference implementation performs this operation through an inline CUDA kernel.


No optimizer is involved.


No gradient update is required.


No model weight is modified.



The Complete Runtime Motor


The current runtime architecture can therefore be written compactly as:


$$
\boxed{
\text{AkbasCore}


\text{Sentetik Pusula Üretimi}
+
\text{SEASC}
+
\text{DRA Zarfı}
+
\text{Kesim Sınırı}
+
\text{İz Gözlemi}
}
$$


or operationally:


SYNTHESIZE DIRECTION
        │
        ▼
MEASURE STATE
        │
        ▼
COMPUTE PHYSICAL DOSE
        │
        ▼
SHAPE DOSE WITH DEPTH
        │
        ▼
INJECT LAYER-LOCAL COMPASS
        │
        ▼
REPEAT L0-L19
        │
        ▼
STOP
        │
        ▼
OBSERVE L20-L27




10. Kesim Sınırı — Intervention Cutoff


AkbasCore deliberately separates active intervention from downstream observation.


The boundary is:


L0 ───────────────────── L19 │ L20 ─────────────────── L27
                            │
      ACTIVE MOTOR          │        MOTOR OFF
      SEASC INJECTION       │      NO INJECTION
                            │
                       KESİM SINIRI



AkbasCore calls this boundary:


Kesim Sınırı — Intervention Cutoff


No steering vector is injected into L20-L27.


This is not a reduced-dose region.


It is not a hidden controller region.


It is not another steering stage.


The motor is off.



11. Sessiz Kuyruk — Silent Tail


The post-intervention transformer region:


L20 → L27



is called:


Sessiz Kuyruk — Silent Tail


Inside the Sessiz Kuyruk:


SEASC injection = 0
steering hooks   = absent
new Pusula dose  = 0



The ordinary transformer continues processing the state.


This creates a clean experimental question:




What does the transformer do with an upstream displacement after the mechanism that created it has stopped?





12. İz Bırakma — Trace Imprinting


The intervention across L0-L19 creates a measurable difference between the steered and Vanilla trajectories.


AkbasCore calls the creation of this upstream downstream-visible difference:


İz Bırakma — Trace Imprinting


This does not imply that the original injected vector survives unchanged.


It means that the active intervention leaves a measurable state difference that later transformer computation can process.



13. İz Taşınımı — Trace Transport


After the motor stops:


$$
\Delta_L


h_L^{SEASC}


h_L^{Vanilla},
\qquad
L\ge20
$$


can continue changing.


AkbasCore calls this:


İz Taşınımı — Trace Transport


The relative displacement can be measured as:


$$
R_L


\frac{
|h_L^{SEASC}-h_L^{Vanilla}|_2
}{
|h_L^{Vanilla}|_2
}
$$


The downstream difference may:


rotate
grow
contract
reorient
change projection



as later transformer blocks process it.


Therefore:


$$
\text{injected vector}
\neq
\text{downstream net displacement}
$$



14. Taşınım İzi — Transport Trace


The ordered sequence:


$$
\Delta_{20},
\Delta_{21},
\dots,
\Delta_{27}
$$


is called:


Taşınım İzi — Transport Trace


It is the measured trajectory of the intervention-created difference through the Sessiz Kuyruk.


Current experiments indicate that these trajectories can be:




structured;


prompt-replicable under controlled conditions;


concept-dependent;


non-universal.




No single universal semantic transport channel has been established.



KIZILELMA — Live Intervention X-Ray


KIZILELMA makes the entire motor visible.


▶ OPEN KIZILELMA LIVE X-RAY


It exposes:


what examples created the contrast
        │
        ▼
what synthetic compasses were built
        │
        ▼
what values exist inside the 20 × 3584 bundle
        │
        ▼
how much was injected
        │
        ▼
where the motor stopped
        │
        ▼
what happened after it stopped
        │
        ▼
whether the generated output changed



The demonstrator records:




matched semantic carriers;


20 real layer-local Pusulas;


71,680 compass values;


requested layer dose;


measured injection magnitude;


Vanilla hidden states;


SEASC hidden states;


28-layer relative displacement;


the L19/L20 cutoff;


Vanilla generated text;


SEASC generated text;


a weight-change sentinel.




It produces:


Overview X-Ray


Designed for immediate visual understanding:


EXTRACT → BUILD → INJECT → STOP → OBSERVE



Technical X-Ray


Designed for technical inspection:


20 × 3584 compass fingerprint
+
measured injection magnitude
+
28-layer Vanilla→SEASC displacement
+
locked envelope metadata




Why KIZILELMA Matters


Generated text alone can hide internal intervention effects.


In a recent reference demonstration:


Vanilla:
The bird is perched on the branch.

SEASC:
The bird is perched on the branch.



The greedy decoded text remained identical.


But the internal trajectories did not.


The measured Vanilla→SEASC relative displacement was already substantial near the end of active steering and continued evolving after the motor stopped, reaching approximately the mid-40% range by L27 in that individual run.


This is not evidence that every run behaves this way.


It demonstrates a more basic methodological point:




No text change does not imply no internal intervention effect.




And conversely:




A large hidden-state displacement does not automatically imply behavioral success.




KIZILELMA therefore displays both.



Hidden-State X-Ray


This distinction first became particularly clear in TEST 142.


For every layer:


$$
\Delta_L


h_L^{SEASC}


h_L^{Vanilla}
$$


Representative TEST 142 measurements:


Mean relative displacement ≈ 30.37%
Mean cos(Δ,A)              ≈ 0.452

L00 relative displacement  ≈ 10%
L19 relative displacement  ≈ 36.90%



The result established an important distinction inside the AkbasCore methodology:


u = intentional injection
Δ = downstream net difference

u ≠ Δ



A small controlled upstream intervention can be transformed substantially by later computation.



Downstream Transport Research


Once the intervention and observation regions were separated, a new question became possible:




Can the motor-OFF evolution of an intervention trace be predicted from measured local geometry?




TEST 192-197 investigated this question.



15. Yerel Taşınım İşleci — Local Transport Operator


Low-rank mappings were estimated between neighboring downstream displacement states.


Conceptually:


$$
\widehat{\Delta}_{L+1}


T_L\Delta_L
$$


AkbasCore calls the experimental mapping:


Yerel Taşınım İşleci — Local Transport Operator


These operators are:


local
empirical
low-rank
support-dependent



They are not claimed to be the full transformer Jacobian.


They do not recover complete 3584-dimensional transformer dynamics.



16. Taşınım Zinciri — Transport Chain


Local predictions can be recursively composed:


$$
\widehat{\Delta}_{L+2}


T_{L+1}T_L\Delta_L
$$


AkbasCore calls this recursive sequence:


Taşınım Zinciri — Transport Chain


This creates two different evaluation regimes:


Teacher-Forced Prediction
        versus
Recursive Free-Run Prediction



The second is harder because prediction errors can propagate through the chain.



17. İnşa Uzayı — BUILD Span


The controlled low-dimensional support used to estimate transport operators is called:


İnşa Uzayı — BUILD Span


For target displacement d and BUILD-support projector P_B:


$$
\operatorname{Coverage}(d)


\frac{
|P_Bd|_2^2
}{
|d|_2^2
}
$$


The BUILD span defines what geometric support is available to the low-rank predictor.



18. Kapsama — Span Coverage


Kapsama — Span Coverage


measures how much of a target displacement is represented by the available BUILD support.


Kapsama is a geometric measurement.


It is not itself a causal variable.


TEST 196 found a strong association between increases in coverage and increases in novel-surface prediction quality.


That association motivated TEST 197.



19. İnşa Kapasitesi — BUILD Capacity


İnşa Kapasitesi — BUILD Capacity


is the controlled size of the support used to construct the İnşa Uzayı.


TEST 197 increased nested BUILD support:


N08
N16
N24
N32



while keeping the SURFACE evaluation fixed.


Results:




BUILD N
Coverage
Teacher
Free L27
Free Error




8
0.684935
0.669637
0.664327
0.740958


16
0.708929
0.692085
0.667969
0.738735


24
0.722478
0.704334
0.673609
0.733351


32
0.735142
0.716287
0.678708
0.727619




Across pooled samples:


Δcoverage → Δprediction

Pearson  = +0.869363
Spearman = +0.816145
N        = 320



Coverage and teacher-forced prediction improved monotonically with BUILD support.


Recursive terminal behavior was more heterogeneous.


Terminal N32−N8:


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



Recorded decision:


BUILD_SPAN_CAPACITY_SCALING_PARTIALLY_REPLICATED



The narrower supported result is:




Increasing BUILD support consistently increased geometric coverage and teacher-forced prediction quality, while recursive terminal improvement remained semantic-axis dependent.





Composite Compass Foundry Research


AkbasCore also developed a broader compass-construction line.


This remains part of the research lineage.



20. Banka Pusulası — Bank-Mean Compass


For matched activation pair i:


$$
d_{L,i}


\operatorname{normalize}
(h^+{L,i}-h^-{L,i})
$$


and:


$$
B_L


\operatorname{normalize}
\left(
\sum_i d_{L,i}
\right)
$$


AkbasCore calls this:


Banka Pusulası — Bank-Mean Compass



21. Subspace-Fisher Reference


A reliability-weighted reference direction was constructed using within-class residual geometry and a regularized low-rank inverse action.


Conceptually:


$$
F_L


\operatorname{normalize}
\left[
(C_L+\lambda_LI)^{-1}
(\mu_L^+-\mu_L^-)
\right]
$$


Subspace-Fisher uses standard mathematical machinery.


AkbasCore does not claim Fisher/LDA geometry, covariance inversion or SVD as project inventions.


Its role here is reference geometry.



22. Artık Yön — Residual Direction


The Banka Pusulası can be decomposed relative to the Fisher reference.


$$
c_L


\langle F_L,B_L\rangle
$$


$$
R_L


\operatorname{normalize}
(B_L-c_LF_L)
$$


AkbasCore calls R_L:


Artık Yön — Residual Direction



23. Bileşik Pusula — Composite Compass


The experimental compass family is:


$$
A_L(\beta)


\operatorname{normalize}
\left[
c_LF_L
+
\beta
\sqrt{1-c_L^2}R_L
\right]
$$


where \beta is:


Bileşim Katsayısı — Composition Coefficient


Under the reference orientation:


β = 0 → Fisher reference
β = 1 → Bank-Mean



The AkbasCore 3.2 reference lock used:


β = 1.0



Representative TEST 162 measurements:




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




This did not establish universal Bank-Mean superiority.


It showed a dose-dependent functional difference under the tested conditions.



Runtime-to-Weight Research


The AkbasCore research program also investigated whether runtime geometry could be partially retained in parameters.


This is not part of the current runtime SEASC motor.



24. Kristalleşme — Crystallization


Kristalleşme — Crystallization


is the AkbasCore term for experiments attempting to transfer part of a runtime steering trajectory into parameter updates and then evaluate with the runtime motor switched off.


TEST 144 used:


LoRA rank  : 8
alpha      : 16
targets    : q/k/v/o
trainable  : ~0.0662%



Final held-out progress was approximately:


+15.79%



under that experiment.


This demonstrated partial retention.


It did not establish complete conversion of runtime steering into weights.



25. Koruma Kalkanı — Retention Guard


Sequential crystallization revealed parameter-space interference.


Koruma Kalkanı — Retention Guard


is the AkbasCore experimental mechanism for projecting a proposed optimizer displacement away from measured retention-sensitive parameter directions.


For displacement d and guard g:


$$
\alpha


\frac{d\cdot g}{g\cdot g}
$$


when the guarded component is active.


The important conclusion was:




Activation-space independence does not imply parameter-space independence.




Retention Guard belongs to the parameter-space research line.


It is not part of the current runtime SEASC motor.



Canonical AkbasCore Terminology




Turkish canonical term
English gloss
AkbasCore meaning




Karşıtlık Ocağı
Contrast Forge
Extracts matched model-internal activation contrasts


Sentetik Pusula Üretimi
Synthetic Compass Synthesis
Synthesizes steering directions from model activations


Pusula
Compass
Unit intervention direction


Katman Pusulası
Layer-Local Compass
Independently synthesized compass for one transformer layer


Pusula Demeti
Compass Bundle
Complete family of layer-local Pusulas


Pusula Dökümhanesi
Compass Foundry
Umbrella direction-construction architecture


Banka Pusulası
Bank-Mean Compass
Aggregated matched activation-pair direction


Artık Yön
Residual Direction
Bank component orthogonal to Fisher reference


Bileşik Pusula
Composite Compass
Fisher/Bank compass family


Bileşim Katsayısı
Composition Coefficient
β control inside Composite Compass


Dondurulmuş Norm
Frozen Norm
Hidden-state norm measured before intervention


Fiziksel Doz
Physical Dose
Intervention relative to frozen state norm


DRA Zarfı
DRA Envelope
Depth-dependent dose profile


İvme
Steering Dose
Initial physical dose


Sönüm
Decay Rate
DRA decay parameter


Zirve
Amplitude
DRA envelope amplitude


Taban
Non-Zero Floor
DRA residual floor


Norma Bağlı Pusula Enjeksiyonu
Norm-Anchored Compass Injection
Direct combination of layer compass, frozen norm and physical dose


Kesim Sınırı
Intervention Cutoff
Exact boundary where active steering ends


Sessiz Kuyruk
Silent Tail
Motor-OFF downstream transformer region


İz Bırakma
Trace Imprinting
Creation of downstream-visible intervention difference


İz Taşınımı
Trace Transport
Downstream evolution after steering stops


Taşınım İzi
Transport Trace
Ordered motor-OFF displacement trajectory


Yerel Taşınım İşleci
Local Transport Operator
Low-rank local displacement predictor


Taşınım Zinciri
Transport Chain
Recursive composition of local predictions


İnşa Uzayı
BUILD Span
Controlled low-dimensional support span


Kapsama
Span Coverage
Fraction represented by BUILD support


İnşa Kapasitesi
BUILD Capacity
Controlled BUILD support size


Kristalleşme
Crystallization
Runtime geometry → parameter retention experiment


Koruma Kalkanı
Retention Guard
Parameter-space interference-control mechanism


KIZILELMA
Live Intervention X-Ray
Public intervention and observation demonstrator


DRA
Damped Resonance Alignment
Depth-envelope research lineage


SEASC
Standardized Evidential Activation-Steering Core
Frozen-norm physical intervention core




These names identify AkbasCore architectural components, experimental abstractions and measurements.


Standard mathematical operations retain their standard scientific provenance.



What Is Actually Running?


The current KIZILELMA runtime path is intentionally much smaller than the complete research tree:


CURRENT RUNTIME

Karşıtlık Ocağı
      │
      ▼
Sentetik Pusula Üretimi
      │
      ▼
20 Katman Pusulası
      │
      ▼
Pusula Demeti
      │
      ▼
DRA Zarfı
      │
      ▼
Dondurulmuş Norm
      │
      ▼
Fiziksel Doz
      │
      ▼
Norma Bağlı Pusula Enjeksiyonu
      │
      ▼
L0-L19
      │
      ▼
Kesim Sınırı
      │
      ▼
Sessiz Kuyruk
L20-L27
      │
      ▼
İz Taşınımı
      │
      ▼
KIZILELMA X-RAY



It does not silently run:


Fisher controller
Bank controller
cosine gate
dynamic brake
saturation controller
runtime cap
lens controller
radar
JVP controller
transport operator
BUILD predictor
Retention Guard
LoRA
weight modification
dynamic feedback controller



Those belong to research history or experimental measurement branches.



Reference Configuration


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




Experimental Lineage


DRA
 │
 ▼
Closed-Loop Activation Steering
 │
 ▼
Synthetic Concept Experiments
 │
 ▼
SEASC
Measured Physical Dose
 │
 ▼
Hidden-State X-Ray
 │
 ▼
Kristalleşme
 │
 ▼
Cross-Vector Interference
 │
 ▼
Koruma Kalkanı
Retention Guard
 │
 ▼
Natural / Causal Transport
 │
 ▼
Semantic Transport
 │
 ▼
Fisher ↔ Bank Decomposition
 │
 ▼
Motor-OFF Tail Experiments
 │
 ▼
İz Taşınımı
 │
 ▼
Yerel Taşınım İşleçleri
 │
 ▼
Novel-Surface Transfer
 │
 ▼
İnşa Uzayı
 │
 ▼
Kapsama
 │
 ▼
İnşa Kapasitesi
TEST 197
 │
 ▼
KIZILELMA
LIVE X-RAY



Negative results remain part of the record.


Mechanisms are not retained merely because they once existed.



What the Evidence Currently Supports


AkbasCore can synthesize layer-local steering directions from model activations


The current implementation constructs twenty independent 3,584-dimensional Pusulas from matched internal activation contrasts.


Physical intervention magnitude is measurable


SEASC expresses requested displacement relative to the pre-intervention hidden-state norm.


Depth and direction are separated


Pusula determines direction.


DRA determines the depth profile.


SEASC determines physical intervention.


Injection and downstream displacement are not the same object


Later transformer computation can rotate and alter an upstream perturbation.


The intervention can be stopped cleanly


L20-L27 can be observed without new SEASC injection.


Motor-OFF traces contain measurable structure


Controlled experiments found prompt- and concept-dependent downstream trajectories.


Low-rank local transport prediction is possible under controlled support


BUILD-span operators predict part of the downstream trace.


Some transport prediction transfers to novel surface formulations


Prediction is not entirely confined to exact BUILD wording.


BUILD support coverage is strongly associated with prediction quality


TEST 196-197 measured a strong relationship under the tested setup.


More support does not universally solve recursive prediction


TEST 197 was deliberately recorded as partially replicated.


Internal change and output change are different measurements


A strong hidden-state displacement can coexist with an unchanged greedy output.



What Is Not Claimed


AkbasCore does not currently establish:




that contrastive steering vectors themselves were invented by AkbasCore;


that difference-of-means activation geometry is a new mathematical technique;


a universal semantic representation;


a universal transformer coordinate system;


a universal full-rank transport operator;


recovery of the complete transformer Jacobian;


complete prediction of transformer dynamics;


universal superiority of one compass-construction method;


universal monotonic benefit from BUILD capacity;


formal causal identification from a CAUSE↔EFFECT semantic axis;


prompt-independent semantic control;


cross-model generalization;


cross-language generalization;


a universal AI alignment solution;


safety guarantees;


human-level hidden-state interpretability.




The strongest current evidence is concentrated on controlled experiments with Qwen2.5-7B-Instruct.


Broader replication remains open.



Reproducibility Philosophy


AkbasCore follows one central experimental rule:




Measure the mechanism separately from the generated text.




Therefore the research distinguishes:


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



A convincing text output is not sufficient evidence.


A large hidden-state displacement is not sufficient evidence of behavioral success.


A correlation is not automatically treated as causation.


A low-rank predictor is not automatically called the full system dynamics.


Negative and falsifying results are preserved.



Project Status


AkbasCore is currently centered on a deliberately measurable motor:


MODEL SYNTHESIZES ITS OWN
LAYER-LOCAL COMPASSES
            │
            ▼
SEASC MEASURES
PHYSICAL INTERVENTION
            │
            ▼
DRA SHAPES
INTERVENTION THROUGH DEPTH
            │
            ▼
CUDA APPLIES
THE INTERVENTION
            │
            ▼
THE MOTOR STOPS
            │
            ▼
THE TRANSFORMER CONTINUES
            │
            ▼
KIZILELMA OBSERVES
WHAT HAPPENS NEXT



The immediate research direction is not to add another controller.


The open questions now concern:


synthetic compass stability
semantic-axis generalization
downstream transport
support geometry
transport capacity
novel-surface transfer
model scale
model family
language
dose regime




Current AkbasCore Research Tree


AKBASCORE
│
├── PUSULA DÖKÜMHANESİ
│   │
│   ├── Karşıtlık Ocağı
│   ├── Sentetik Pusula Üretimi
│   ├── Katman Pusulası
│   ├── Pusula Demeti
│   │
│   ├── Banka Pusulası        [research]
│   ├── Subspace-Fisher       [reference]
│   ├── Artık Yön             [research]
│   └── Bileşik Pusula        [research]
│
├── SEASC
│   │
│   ├── Dondurulmuş Norm
│   ├── Fiziksel Doz
│   └── Norma Bağlı Pusula Enjeksiyonu
│
├── DRA ZARFI
│   │
│   ├── İvme
│   ├── Sönüm
│   ├── Zirve
│   └── Taban
│
├── MÜDAHALE SINIRI
│   │
│   ├── Kesim Sınırı
│   └── Sessiz Kuyruk
│
├── TAŞINIM ARAŞTIRMASI
│   │
│   ├── İz Bırakma
│   ├── İz Taşınımı
│   ├── Taşınım İzi
│   ├── Yerel Taşınım İşleci
│   ├── Taşınım Zinciri
│   ├── İnşa Uzayı
│   ├── Kapsama
│   └── İnşa Kapasitesi
│
├── PARAMETRE-UZAYI ARAŞTIRMASI
│   │
│   ├── Kristalleşme
│   └── Koruma Kalkanı
│
└── KIZILELMA
    │
    ├── Overview X-Ray
    └── Technical X-Ray





The model provides the geometry.

The Foundry synthesizes the compasses.

SEASC measures the dose.

DRA shapes the depth.

CUDA applies the intervention.

The motor stops.

KIZILELMA watches the trace.

The weights remain frozen.





Citation


For scientific or technical work based materially on AkbasCore, please cite the relevant versioned AkbasCore release and associated archival record rather than treating the continuously updated main branch as a fixed publication.


The repository's timestamped commits, Releases, experiment records and archival deposits preserve the development lineage.


Project: AkbasCore

Developer / Researcher: Mustafa Akbaş

Research lineage: DRA → SEASC → Transport → KIZILELMA

Year: 2026



License


AkbasCore software is currently distributed under the license terms provided in this repository.


Copyright © 2026 Mustafa Akbaş — AkbasCore


The repository contains software together with experimental methodology, research results, architectural terminology and attribution information.


For the complete and controlling terms, scope, copyright notice and attribution information, read the repository's LICENSE file.


Standard mathematical techniques used within AkbasCore retain their existing scientific provenance.



AkbasCore — SEASC


Mustafa Akbaş · 2026




Direction is synthesized.

Dose is measured.

Depth is shaped.

The motor stops.

The trace remains visible.




SEE THE INTERVENTION, NOT JUST THE OUTPUT.


▶ KIZILELMA LIVE X-RAY

