# =============================================================================
# TEST 151 v2 — SYNTHETIC FOUNDRY — LOCKED
# MODEL-NATIVE SYNTHETIC DIRECTION FALSIFICATION ASSAY
# AkbasCore · Qwen2.5-7B-Instruct · A100
# =============================================================================

import sys, subprocess, os, gc, warnings
for p,n in [("transformers>=4.46.0","transformers"),("accelerate","accelerate")]:
    try: __import__(n)
    except: subprocess.check_call([sys.executable,"-m","pip","install","-q",p])

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"]="false"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU bulunamadı.")

SEED=42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

DEVICE="cuda"
MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N_LAYERS=20
SOURCE_LAYER=3
TARGET_LAYERS=list(range(SOURCE_LAYER+1,N_LAYERS))

EPS_REL=0.0025
N_CANDIDATES=32
N_NULL_TOURNAMENTS=16
SYSTEM="You are a concise reasoning assistant."

A_POS=[
    "a person standing above a bridge",
    "a person positioned above a tower",
    "an object located above the roof",
    "a bird flying above the mountain",
    "a light appearing above the building",
    "a person looking from above the valley",
    "a structure extending above the wall",
    "a signal coming from above the surface",
]
A_NEG=[
    "a person standing below a bridge",
    "a person positioned below a tower",
    "an object located below the roof",
    "a bird flying below the mountain",
    "a light appearing below the building",
    "a person looking from below the valley",
    "a structure extending below the wall",
    "a signal coming from below the surface",
]
B_POS=[
    "an alarm sounded before the meeting",
    "the light appeared before the noise",
    "the door opened before the bell rang",
    "the bird landed before the rain started",
    "the message arrived before the event",
    "the engine stopped before the signal",
    "the person left before the announcement",
    "the screen changed before the sound",
]
B_NEG=[
    "an alarm sounded after the meeting",
    "the light appeared after the noise",
    "the door opened after the bell rang",
    "the bird landed after the rain started",
    "the message arrived after the event",
    "the engine stopped after the signal",
    "the person left after the announcement",
    "the screen changed after the sound",
]

OBJECTS=[
    "ceramic cup","closed notebook","metal spoon","wooden bowl",
    "glass bottle","blue folder","black pen","small lamp",
    "cardboard box","folded newspaper","desk clock","paper envelope",
    "wooden ruler","white plate","cloth bag","plastic container"
]
PLACES=["desk","table","counter","shelf"]
NEUTRAL=[
    f"A {OBJECTS[i%16]} and a {OBJECTS[(i*5+3)%16]} are visible on a {PLACES[(i//16)%4]}."
    for i in range(64)
]

PROBES=[
    "A wooden bowl contains three smooth stones.",
    "A person reads a short article in a quiet room.",
    "Several tools are arranged on a workbench.",
    "A blue folder lies beside a black pen.",
]

print("="*112)
print("TEST 151 v2 — SYNTHETIC FOUNDRY — LOCKED")
print("MODEL-NATIVE SYNTHETIC DIRECTION FALSIFICATION ASSAY")
print("="*112)

# =============================================================================
# 1. MODEL
# =============================================================================

tok=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tok.pad_token_id is None: tok.pad_token=tok.eos_token

model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map={"":0},
    attn_implementation="sdpa",
    trust_remote_code=True,
)
model.eval()
for p in model.parameters(): p.requires_grad_(False)

layers=model.model.layers
H=model.config.hidden_size

print("GPU              :",torch.cuda.get_device_name(0))
print("Model            :",MODEL_ID)
print("Hidden           :",H)
print("Source           :",f"L{SOURCE_LAYER}")
print("Targets          :",f"L{TARGET_LAYERS[0]}-L{TARGET_LAYERS[-1]}")
print("Dose             :",f"{100*EPS_REL:.4f}%")
print("Candidates       :",N_CANDIDATES)
print("Null tournaments :",N_NULL_TOURNAMENTS)

# =============================================================================
# 2. ACTIVATIONS
# =============================================================================

@torch.inference_mode()
def states(texts):
    out=[[] for _ in range(N_LAYERS)]
    for text in texts:
        x=tok(text,return_tensors="pt",add_special_tokens=True).to(DEVICE)
        y=model(**x,output_hidden_states=True,use_cache=False,return_dict=True)
        pos=int(x["attention_mask"][0].sum().item())-1
        for L in range(N_LAYERS):
            out[L].append(y.hidden_states[L+1][0,pos].float().detach())
        del x,y
    return [torch.stack(v) for v in out]

print("\n[1/6] Parent A...")
AP=states(A_POS); AN=states(A_NEG)

print("[2/6] Parent B...")
BP=states(B_POS); BN=states(B_NEG)

print("[3/6] Neutral residual bank...")
N=states(NEUTRAL)

A=[]; B=[]
raw_parent_cos=[]

for L in range(N_LAYERS):
    ar=AP[L].mean(0)-AN[L].mean(0)
    br=BP[L].mean(0)-BN[L].mean(0)

    an=F.normalize(ar,dim=0)
    bn0=F.normalize(br,dim=0)
    raw_parent_cos.append(float(torch.dot(an,bn0).item()))

    br=br-torch.dot(br,an)*an
    bn=F.normalize(br,dim=0)

    A.append(an)
    B.append(bn)

print(f"Raw A/B cosine @ L{SOURCE_LAYER}: {raw_parent_cos[SOURCE_LAYER]:+.6f}")

del AP,AN,BP,BN
gc.collect()
torch.cuda.empty_cache()

# =============================================================================
# 3. FOUNDRY
# =============================================================================

bank=N[SOURCE_LAYER]
mu=bank.mean(0)
R=bank-mu
rank=int(torch.linalg.matrix_rank(R).item())

def remove_parent_span(z):
    z=z-torch.dot(z,A[SOURCE_LAYER])*A[SOURCE_LAYER]
    z=z-torch.dot(z,B[SOURCE_LAYER])*B[SOURCE_LAYER]
    n=z.norm()
    if n<1e-10: return None
    return z/n

def parent_overlap(z):
    return max(
        abs(float(torch.dot(z,A[SOURCE_LAYER]).item())),
        abs(float(torch.dot(z,B[SOURCE_LAYER]).item()))
    )

candidates=[]

for i in range(N_CANDIDATES):
    w=torch.randn(R.shape[0],device=DEVICE)
    w=w-w.mean()
    raw=(w[:,None]*R).sum(0)

    raw_unit=F.normalize(raw,dim=0)
    raw_overlap=parent_overlap(raw_unit)

    z=remove_parent_span(raw)
    if z is None: continue

    candidates.append({
        "id":i,
        "z":z,
        "raw_parent_overlap":raw_overlap,
        "post_parent_overlap":parent_overlap(z)
    })

print("Neutral samples   :",len(NEUTRAL))
print("Residual rank     :",rank)
print("Valid candidates  :",len(candidates))

# =============================================================================
# 4. TRANSPORT
# =============================================================================

@torch.inference_mode()
def transport(prompt,z):

    text=tok.apply_chat_template(
        [{"role":"system","content":SYSTEM},
         {"role":"user","content":prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    x=tok(text,return_tensors="pt").to(DEVICE)
    pos=int(x["attention_mask"][0].sum().item())-1

    base=[None]*N_LAYERS
    pert=[None]*N_LAYERS
    handles=[]

    def capture(store,L):
        def hook(m,a,o):
            h=o[0] if isinstance(o,tuple) else o
            store[L]=h[0,pos].float().detach().clone()
        return hook

    for L in range(N_LAYERS):
        handles.append(layers[L].register_forward_hook(capture(base,L)))

    model(**x,use_cache=False,return_dict=True)

    for h in handles: h.remove()
    handles=[]

    source_norm=base[SOURCE_LAYER].norm().detach()

    def inject(m,a,o):
        h=o[0] if isinstance(o,tuple) else o
        q=h.clone()
        delta=EPS_REL*q[0,pos].float().norm()*z
        q[0,pos]=(q[0,pos].float()+delta).to(q.dtype)
        return (q,)+o[1:] if isinstance(o,tuple) else q

    for L in range(N_LAYERS):
        if L==SOURCE_LAYER:
            handles.append(layers[L].register_forward_hook(inject))
        handles.append(layers[L].register_forward_hook(capture(pert,L)))

    model(**x,use_cache=False,return_dict=True)

    for h in handles: h.remove()

    injected_norm=float((EPS_REL*source_norm).item())

    gains=[]
    local_cos=[]
    source_cos=[]
    prev=z

    for L in TARGET_LAYERS:
        d=pert[L]-base[L]
        dn=float(d.norm().item())

        gains.append(dn/max(injected_norm,1e-12))

        if dn>1e-8:
            du=d/d.norm()
            local_cos.append(float(torch.dot(du,prev).item()))
            source_cos.append(float(torch.dot(du,z).item()))
            prev=du
        else:
            local_cos.append(0.0)
            source_cos.append(0.0)

    del x
    return np.asarray(gains),np.asarray(local_cos),np.asarray(source_cos)

# =============================================================================
# 5. SCORE
# =============================================================================

def score_direction(z):

    G=[]; LC=[]; SC=[]

    for p in PROBES:
        g,l,s=transport(p,z)
        G.append(g); LC.append(l); SC.append(s)

    G=np.stack(G)
    LC=np.stack(LC)
    SC=np.stack(SC)

    prompt_gain=G.mean(axis=1)
    endpoint=G[:,-1]

    gain_cv=float(prompt_gain.std()/(abs(prompt_gain.mean())+1e-8))
    endpoint_cv=float(endpoint.std()/(abs(endpoint.mean())+1e-8))

    local=float(LC.mean())
    source=float(SC.mean())

    prompt_stability=1.0/(1.0+gain_cv)
    endpoint_stability=1.0/(1.0+endpoint_cv)

    fitness=max(0.0,local)*prompt_stability*endpoint_stability

    return {
        "fitness":fitness,
        "gain":float(G.mean()),
        "gain_cv":gain_cv,
        "endpoint_cv":endpoint_cv,
        "local":local,
        "source":source,
        "prompt_stability":prompt_stability,
        "endpoint_stability":endpoint_stability,
    }

# =============================================================================
# 6. REAL TOURNAMENT
# =============================================================================

print("\n[4/6] Real residual-span tournament...")

for i,c in enumerate(candidates,1):
    c.update(score_direction(c["z"]))
    if i%8==0: print(f"  {i}/{len(candidates)}")

candidates.sort(key=lambda x:x["fitness"],reverse=True)
best=candidates[0]
REAL_MAX=best["fitness"]

# =============================================================================
# 7. MATCHED NULL TOURNAMENTS
# =============================================================================

print("[5/6] Matched isotropic null tournaments...")

null_max=[]
null_mean=[]

for t in range(N_NULL_TOURNAMENTS):

    fs=[]

    for _ in range(N_CANDIDATES):
        z=None
        while z is None:
            z=remove_parent_span(torch.randn(H,device=DEVICE))

        fs.append(score_direction(z)["fitness"])

    null_max.append(max(fs))
    null_mean.append(float(np.mean(fs)))

    print(
        f"  null {t+1:02d}/{N_NULL_TOURNAMENTS} | "
        f"max={null_max[-1]:.6f}"
    )

null_max=np.asarray(null_max,dtype=np.float64)
null_mean=np.asarray(null_mean,dtype=np.float64)

# =============================================================================
# 8. TOURNAMENT STATISTICS
# =============================================================================

exceed=int(np.sum(null_max>=REAL_MAX))
empirical_p=(exceed+1)/(N_NULL_TOURNAMENTS+1)

nm=float(null_max.mean())
ns=float(null_max.std(ddof=1)) if len(null_max)>1 else 0.0
descriptive_z=(REAL_MAX-nm)/(ns+1e-12)

percentile=100.0*float(np.mean(null_max<REAL_MAX))

real_all=np.asarray([c["fitness"] for c in candidates])
real_mean=float(real_all.mean())

# =============================================================================
# 9. REPORT
# =============================================================================

print("\n"+"="*126)
print("TOP MODEL-NATIVE DIRECTIONS")
print("="*126)
print(
    f"{'ID':>3} {'FITNESS':>10} {'GAIN':>9} {'LOCAL':>9} {'SOURCE':>9} "
    f"{'GAIN_CV':>9} {'END_CV':>9} {'RAW_AB':>9} {'POST_AB':>10}"
)

for c in candidates[:8]:
    print(
        f"{c['id']:>3} "
        f"{c['fitness']:>10.6f} "
        f"{c['gain']:>9.4f} "
        f"{c['local']:>9.4f} "
        f"{c['source']:>9.4f} "
        f"{c['gain_cv']:>9.4f} "
        f"{c['endpoint_cv']:>9.4f} "
        f"{c['raw_parent_overlap']:>9.5f} "
        f"{c['post_parent_overlap']:>10.7f}"
    )

print("\n"+"="*126)
print("TEST 151 v2 — LOCKED FALSIFICATION SUMMARY")
print("="*126)

print(f"Neutral residual rank                : {rank}")
print(f"Real candidate count                 : {len(candidates)}")
print(f"Null tournament size                 : {N_CANDIDATES}")
print(f"Null tournament count                : {N_NULL_TOURNAMENTS}")
print(f"Best real candidate                  : C{best['id']}")
print(f"Best real fitness                    : {REAL_MAX:.8f}")
print(f"Real population mean fitness         : {real_mean:.8f}")
print(f"Best mean downstream gain            : {best['gain']:.8f}")
print(f"Best local transport coherence       : {best['local']:.8f}")
print(f"Best source-direction alignment      : {best['source']:.8f}")
print(f"Best gain CV                         : {best['gain_cv']:.8f}")
print(f"Best endpoint CV                     : {best['endpoint_cv']:.8f}")
print(f"Best raw parent overlap              : {best['raw_parent_overlap']:.8f}")
print(f"Best post-projection parent overlap  : {best['post_parent_overlap']:.8f}")
print(f"Null tournament-max mean             : {nm:.8f}")
print(f"Null tournament-max SD               : {ns:.8f}")
print(f"Null maxima >= real maximum          : {exceed}/{N_NULL_TOURNAMENTS}")
print(f"Empirical tournament p               : {empirical_p:.6f}")
print(f"Real-max percentile vs null maxima   : {percentile:.2f}%")
print(f"Descriptive tournament Z             : {descriptive_z:.4f}")
print(f"Minimum possible empirical p         : {1/(N_NULL_TOURNAMENTS+1):.6f}")

print("\nNULL TOURNAMENT MAXIMA")
print(np.array2string(null_max,precision=6,separator=", "))

print("\n"+"="*126)
print("LOCKED INTERPRETATION")
print("="*126)

if exceed==0:
    print("SCREEN: REAL MAX EXCEEDED ALL NULL TOURNAMENT MAXIMA.")
elif REAL_MAX>nm:
    print("SCREEN: REAL MAX ABOVE NULL-MAX MEAN, BUT NULL OVERLAP EXISTS.")
else:
    print("SCREEN: NO TOURNAMENT-LEVEL SEPARATION FROM ISOTROPIC NULL.")

print("""
This assay does not claim discovery of a concept.
It tests whether A/B-span-excluded directions synthesized from the model's
own neutral residual activation subspace exhibit unusually coherent and
prompt-stable downstream transport relative to equally searched isotropic
random directions.

LOCAL coherence measures smooth layer-to-layer transport:
    cos(delta_L, delta_{L-1})

SOURCE alignment is separate telemetry:
    cos(delta_L, z_source)

Rotation away from the source direction is therefore not automatically
penalized. No semantic label is assigned to the surviving direction.

With 16 null tournaments the minimum empirical p is 1/17; descriptive Z
and percentile are screening statistics, not claims of significance.
""")

print("="*126)
print("TEST 151 v2 COMPLETE")
print("="*126)
