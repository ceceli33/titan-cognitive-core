# ======================================================================================================================
# TEST 181 — TERMINAL REVERSAL CAUSAL ABLATION
# AkbasCore 3.2 / SEASC
#
# Direct continuation of TEST180.
#
# AXIS:
#   E = CAUSE ↔ EFFECT
#
# TEST180 OBSERVATION:
#   E was a previously failed forge.
#   Its late-average target projection remained positive,
#   but L19 target projection reversed sign:
#
#       L17  +1.317
#       L18  +1.142
#       L19  -0.548
#
# QUESTION:
#   Is the terminal reversal merely correlated with behavioral failure,
#   or does changing the terminal intervention alter causal behavior?
#
# FROZEN ARMS:
#   NORMAL      : L00-L19 normal SEASC
#   L19_OFF     : L00-L18 normal, L19 dose = 0
#   L19_FLIP    : L00-L18 normal, L19 injection sign reversed
#   L18_19_OFF  : L00-L17 normal, L18-L19 dose = 0
#
# IMPORTANT:
#   - Same Qwen2.5-7B-Instruct
#   - Same SEASC envelope
#   - Same RSS lineage
#   - Same E Fisher/Bank
#   - Same P8
#   - Same E causal-gradient forge
#   - No vector search
#   - No rank search
#   - No layer search
#   - No envelope search
#   - No controller
#   - No weight modification
#   - Arms are preregistered from TEST180 mechanistic observation
#   - Fresh DEV/FINAL/NEUTRAL prompts
#   - FINAL is not used for selection
# ======================================================================================================================

import os,sys,math,json,csv,random,shutil,subprocess,importlib.util,gc
from pathlib import Path

if importlib.util.find_spec("ninja") is None:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*154)
print("TEST 181 — TERMINAL REVERSAL CAUSAL ABLATION")
print("TEST180 CAUSE↔EFFECT FAILURE — L19 INTERVENTION TEST")
print("="*154)

os.environ["MAX_JOBS"]="2"
os.environ["TOKENIZERS_PARALLELISM"]="false"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")

DEVICE=torch.device("cuda")
SEED=181
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N_LAYERS=20

IVME=.10
SONUM=.30
ZIRVE=.70
TABAN=.20

FISHER_ALPHA=.10
FISHER_EPS=1e-8
EPS=1e-10
SVD_REL_TOL=1e-5

PRIMARY_SCALE=.50

SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."

ROOT=Path("/content/TEST181_TERMINAL_REVERSAL_CAUSAL_ABLATION")
ROOT.mkdir(parents=True,exist_ok=True)

def env(L):
    x=ZIRVE*math.exp(-SONUM*float(L))*(1+SONUM*float(L))+TABAN
    return x/(ZIRVE+TABAN)

ENVELOPE=np.asarray([env(L) for L in range(N_LAYERS)],dtype=np.float64)
RHO=IVME*ENVELOPE
RSS=float(np.sqrt(np.sum(RHO**2)))

print("GPU:",torch.cuda.get_device_name(0))
print("Model:",MODEL_ID)
print("Axis: E CAUSE↔EFFECT")
print(f"Locked SEASC RSS: {RSS:.9f}")
print("Primary scale:",PRIMARY_SCALE)

# ======================================================================================================================
# [1/28] CUDA SEASC
# ======================================================================================================================

BUILD="/tmp/akbascore_test181"
shutil.rmtree(BUILD,ignore_errors=True)
os.makedirs(BUILD,exist_ok=True)

CPP=r"""
#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
TORCH_CHECK(hidden.is_cuda()&&actuator.is_cuda()&&dose.is_cuda(),"CUDA tensors required");
TORCH_CHECK(hidden.dim()==3&&actuator.dim()==2&&dose.dim()==1,"bad dimensions");
TORCH_CHECK(hidden.size(0)==actuator.size(0)&&hidden.size(0)==dose.size(0)&&hidden.size(2)==actuator.size(1),"shape mismatch");
return seasc_batch_cuda(hidden,actuator,dose);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST181 SEASC");}
"""

CUDA=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename scalar_t>
__global__ void k(scalar_t* h,const float* a,const float* d,const int B,const int S,const int H){
int vec=blockIdx.x,b=vec/S;
if(b>=B)return;
extern __shared__ float sh[];
long long base=(long long)vec*H,ab=(long long)b*H;
float ss=0.f;
for(int j=threadIdx.x;j<H;j+=blockDim.x){
    float x=(float)h[base+j];
    ss+=x*x;
}
sh[threadIdx.x]=ss;
__syncthreads();
for(unsigned s=blockDim.x/2;s>0;s>>=1){
    if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];
    __syncthreads();
}
float n=sqrtf(fmaxf(sh[0],1e-20f));
float scale=d[b]*n;
__syncthreads();
for(int j=threadIdx.x;j<H;j+=blockDim.x){
    float x=(float)h[base+j];
    h[base+j]=(scalar_t)(x+scale*a[ab+j]);
}
}

torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
auto out=hidden.contiguous().clone();
auto act=actuator.to(hidden.device(),torch::kFloat32).contiguous();
auto d=dose.to(hidden.device(),torch::kFloat32).contiguous();
int B=out.size(0),S=out.size(1),H=out.size(2),nv=B*S;
constexpr int T=256;
size_t sb=T*sizeof(float);
cudaStream_t stream=at::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    out.scalar_type(),
    "test181_seasc",
    [&]{
        k<scalar_t><<<nv,T,sb,stream>>>(
            out.data_ptr<scalar_t>(),
            act.data_ptr<float>(),
            d.data_ptr<float>(),
            B,S,H);
    });
C10_CUDA_KERNEL_LAUNCH_CHECK();
return out;
}
"""

print("\n[1/28] Compiling frozen SEASC...")
seasc_ext=load_inline(
    name="akbascore_test181_cuda",
    cpp_sources=CPP,
    cuda_sources=CUDA,
    functions=None,
    extra_cflags=["-O3","-std=c++17"],
    extra_cuda_cflags=["-O3","--use_fast_math"],
    with_cuda=True,
    build_directory=BUILD,
    verbose=False)
print("       OK")

# ======================================================================================================================
# [2/28] MODEL
# ======================================================================================================================

print("\n[2/28] Loading frozen model...")

tokenizer=AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token=tokenizer.eos_token

model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map={"":0},
    attn_implementation="sdpa",
    trust_remote_code=True)

model.eval()

for p in model.parameters():
    p.requires_grad_(False)

layers=model.model.layers
HIDDEN=model.config.hidden_size

print("       hidden:",HIDDEN,"dtype:",next(model.parameters()).dtype)

def chat(x):
    return tokenizer.apply_chat_template(
        [
            {"role":"system","content":SYSTEM},
            {"role":"user","content":x}
        ],
        tokenize=False,
        add_generation_prompt=True)

def norm(v):
    return v/v.norm(dim=-1,keepdim=True).clamp_min(EPS)

@torch.inference_mode()
def capture(text):
    e=tokenizer(chat(text),return_tensors="pt").to(DEVICE)
    o=model(
        **e,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True)
    pos=int(e["attention_mask"][0].sum())-1
    hs=[
        o.hidden_states[L+1][0,pos].float().detach()
        for L in range(N_LAYERS)
    ]
    del e,o
    return hs

def sequence(context,continuation):
    p=tokenizer(
        chat(context),
        return_tensors="pt",
        add_special_tokens=False)
    c=tokenizer(
        continuation,
        return_tensors="pt",
        add_special_tokens=False)

    ids=torch.cat(
        [p["input_ids"],c["input_ids"]],
        dim=1).to(DEVICE)

    return ids,torch.ones_like(ids),p["input_ids"].shape[1]

def activation_gradient(context,continuation,positive_score=False):
    ids,mask,plen=sequence(context,continuation)

    emb=model.get_input_embeddings()(ids).detach().requires_grad_(True)

    saved=[None]*N_LAYERS
    hooks=[]

    for L in range(N_LAYERS):
        def mk(li):
            def hk(m,a,o):
                h=o[0] if isinstance(o,tuple) else o
                h.retain_grad()
                saved[li]=h
            return hk
        hooks.append(layers[L].register_forward_hook(mk(L)))

    model.zero_grad(set_to_none=True)

    try:
        o=model(
            inputs_embeds=emb,
            attention_mask=mask,
            use_cache=False,
            return_dict=True)

        z=o.logits[:,plen-1:-1,:].float()
        tar=ids[:,plen:]

        lp=F.log_softmax(z,-1)
        score=lp.gather(
            -1,
            tar.unsqueeze(-1)).squeeze(-1).mean()

        objective=score if positive_score else -score
        objective.backward()

        gg=[]

        for L in range(N_LAYERS):
            g=saved[L].grad[0].float().sum(dim=0)
            gg.append(g.detach().clone())

    finally:
        for h in hooks:
            h.remove()

        model.zero_grad(set_to_none=True)

    return gg

# ======================================================================================================================
# [3/28] E AXIS — SAME TEST180 SEMANTIC PAIRS
# ======================================================================================================================

print("\n[3/28] Loading frozen E CAUSE↔EFFECT axis...")

E_PAIRS=[
("The spark is the cause of the fire.","The fire is the effect of the spark."),
("The impact is the cause of the crack.","The crack is the effect of the impact."),
("The rainfall is the cause of the flooding.","The flooding is the effect of the rainfall."),
("The heat is the cause of the melting.","The melting is the effect of the heat."),
("The collision is the cause of the damage.","The damage is the effect of the collision."),
("The pressure is the cause of the deformation.","The deformation is the effect of the pressure."),
("The infection is the cause of the fever.","The fever is the effect of the infection."),
("The vibration is the cause of the loosening.","The loosening is the effect of the vibration."),
("The frost is the cause of the cracking.","The cracking is the effect of the frost."),
("The blockage is the cause of the overflow.","The overflow is the effect of the blockage."),
("The friction is the cause of the heating.","The heating is the effect of the friction."),
("The force is the cause of the motion.","The motion is the effect of the force."),
("The outage is the cause of the shutdown.","The shutdown is the effect of the outage."),
("The leak is the cause of the pressure loss.","The pressure loss is the effect of the leak."),
("The acceleration is the cause of the displacement.","The displacement is the effect of the acceleration."),
("The trigger is the cause of the response.","The response is the effect of the trigger.")
]

# ======================================================================================================================
# [4/28] E FISHER / BANK
# ======================================================================================================================

print("\n[4/28] Reconstructing E Fisher / Bank...")

PB=[[] for _ in range(N_LAYERS)]
NB=[[] for _ in range(N_LAYERS)]

for i,(p,n) in enumerate(E_PAIRS,1):
    hp=capture(p)
    hn=capture(n)

    for L in range(N_LAYERS):
        PB[L].append(hp[L])
        NB[L].append(hn[L])

    print(f"       E pair {i:02d}/16")

PB=[torch.stack(x).float() for x in PB]
NB=[torch.stack(x).float() for x in NB]

E_FISHER=[]
E_BANK=[]
FB_COS=[]
FB_RES=[]

for L in range(N_LAYERS):
    Hp,Hn=PB[L],NB[L]

    mp=Hp.mean(0)
    mn=Hn.mean(0)
    md=mp-mn

    RR=torch.cat([Hp-mp,Hn-mn],0).float()
    nu=max(RR.shape[0]-2,1)

    _,S,Vh=torch.linalg.svd(
        RR,
        full_matrices=False)

    eig=S.square()/float(nu)

    pe=eig[eig>FISHER_EPS]

    es=(
        torch.median(pe)
        if pe.numel()
        else torch.tensor(1.,device=DEVICE)
    )

    lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS)

    V=Vh.T
    pr=V.T@md

    f=norm(
        md/lam+
        V@((1/(eig+lam)-1/lam)*pr)
    ).float()

    dirs=norm(Hp-Hn)

    dirs=dirs*torch.where(
        (dirs@f)[:,None]<0,
        -torch.ones_like(dirs[:,:1]),
        torch.ones_like(dirs[:,:1])
    )

    b=norm(dirs.mean(0)).float()

    c=torch.dot(f,b).clamp(-1,1)

    if c<0:
        b=-b
        c=-c

    E_FISHER.append(f)
    E_BANK.append(b)
    FB_COS.append(float(c))
    FB_RES.append(float((b-c*f).norm()))

E_FISHER=torch.stack(E_FISHER).contiguous()
E_BANK=torch.stack(E_BANK).contiguous()

print(
    f"       E mean cos(F,B)={np.mean(FB_COS):+.6f} "
    f"residual={np.mean(FB_RES):.6f}"
)

# ======================================================================================================================
# [5/28] SAME FROZEN P8
# ======================================================================================================================

print("\n[5/28] Building shared frozen P8...")

RETENTION_BUILD=[
("A ceramic cup rests beside a folded newspaper."," The scene contains ordinary stationary objects."),
("A metal ruler lies beside a closed notebook."," The scene contains ordinary stationary objects."),
("A wooden bowl sits near a plain cloth."," The scene contains ordinary stationary objects."),
("A glass jar rests beside a cardboard package."," The scene contains ordinary stationary objects."),
("A calculator lies next to a paper envelope."," The objects remain stationary."),
("A closed umbrella rests beside a travel bag."," The objects remain stationary."),
("A spoon lies beside an empty plate."," The objects remain stationary."),
("A pencil rests next to a blank sheet of paper."," The objects remain stationary.")
]

PG=[]

for i,(ctx,cont) in enumerate(RETENTION_BUILD,1):
    PG.append(
        activation_gradient(
            ctx,
            cont,
            positive_score=False))
    print(f"       probe {i}/8")

Q8=[]
RANKS=[]

for L in range(N_LAYERS):
    G=torch.stack(
        [PG[i][L] for i in range(8)]
    ).float()

    G=G[G.norm(dim=1)>EPS]
    G=norm(G)

    _,s,Vh=torch.linalg.svd(
        G,
        full_matrices=False)

    tol=max(
        float(s[0])*SVD_REL_TOL,
        EPS)

    rank=int((s>tol).sum())

    Q8.append(
        Vh[:rank].T.contiguous())

    RANKS.append(rank)

print("       ranks:",RANKS)

def project_basis(raw,bases):
    out=[]

    for L in range(N_LAYERS):
        u=raw[L].float()
        q=bases[L]

        if q.shape[1]:
            v=u-q@(q.T@u)
        else:
            v=u

        if v.norm()<EPS:
            raise RuntimeError(
                f"Projection collapsed L{L}")

        out.append(norm(v))

    return torch.stack(out).contiguous()

E_FISHER_GUARD=project_basis(
    E_FISHER,
    Q8)

# ======================================================================================================================
# [6/28] SAME E CAUSAL FORGE
# ======================================================================================================================

print("\n[6/28] Loading frozen E causal-gradient forge...")

E_FORGE={

"A":[
("A spark and a fire are related causally. Which item is the initiating event?",
 " The spark is the cause.",
 " The fire is the effect."),

("An impact and a crack are related causally. Which item is the initiating event?",
 " The impact is the cause.",
 " The crack is the effect."),

("Rainfall and flooding are related causally. Which item is the initiating event?",
 " The rainfall is the cause.",
 " The flooding is the effect."),

("Heat and melting are related causally. Which item is the initiating event?",
 " The heat is the cause.",
 " The melting is the effect.")
],

"B":[
("A collision produces damage. Identify the causal role of the collision rather than its consequence.",
 " The collision produces the damage.",
 " The damage results from the collision."),

("Pressure produces deformation. Identify the causal role of the pressure rather than its consequence.",
 " The pressure produces the deformation.",
 " The deformation results from the pressure."),

("An infection produces fever. Identify the causal role of the infection rather than its consequence.",
 " The infection produces the fever.",
 " The fever results from the infection."),

("Vibration produces loosening. Identify the causal role of the vibration rather than its consequence.",
 " The vibration produces the loosening.",
 " The loosening results from the vibration.")
],

"C":[
("Consider the directional causal relation between frost and cracking.",
 " Causally, frost precedes and generates the cracking.",
 " Causally, cracking follows from the frost."),

("Consider the directional causal relation between blockage and overflow.",
 " Causally, blockage precedes and generates the overflow.",
 " Causally, overflow follows from the blockage."),

("Consider the directional causal relation between friction and heating.",
 " Causally, friction precedes and generates the heating.",
 " Causally, heating follows from the friction."),

("Consider the directional causal relation between force and motion.",
 " Causally, force precedes and generates the motion.",
 " Causally, motion follows from the force.")
],

"D":[
("An outage and a shutdown form a cause-effect pair. Select the upstream member.",
 " The outage is upstream in the causal chain.",
 " The shutdown is downstream in the causal chain."),

("A leak and pressure loss form a cause-effect pair. Select the upstream member.",
 " The leak is upstream in the causal chain.",
 " The pressure loss is downstream in the causal chain."),

("Acceleration and displacement form a cause-effect pair. Select the upstream member.",
 " The acceleration is upstream in the causal chain.",
 " The displacement is downstream in the causal chain."),

("A trigger and a response form a cause-effect pair. Select the upstream member.",
 " The trigger is upstream in the causal chain.",
 " The response is downstream in the causal chain.")
]
}

# ======================================================================================================================
# [7/28] RECONSTRUCT E CAUSAL ORTH
# ======================================================================================================================

print("\n[7/28] Reconstructing frozen E causal vector...")

E_FAMILIES={}

for fn,items in E_FORGE.items():

    acc=[
        torch.zeros(
            HIDDEN,
            device=DEVICE,
            dtype=torch.float32)
        for _ in range(N_LAYERS)
    ]

    for j,(ctx,pos,neg) in enumerate(items,1):

        gp=activation_gradient(
            ctx,
            pos,
            positive_score=True)

        gn=activation_gradient(
            ctx,
            neg,
            positive_score=True)

        for L in range(N_LAYERS):
            acc[L]+=gp[L]-gn[L]

        print(
            f"       E family {fn} {j}/4")

    v=torch.stack(
        [norm(x) for x in acc]
    ).contiguous()

    for L in range(N_LAYERS):
        if torch.dot(
            v[L],
            E_FISHER[L])<0:
            v[L]=-v[L]

    E_FAMILIES[fn]=v

E_CAUSAL=torch.stack([
    norm(
        torch.stack([
            E_FAMILIES[f][L]
            for f in E_FAMILIES
        ]).mean(0)
    )
    for L in range(N_LAYERS)
]).contiguous()

E_CAUSAL_GUARD=project_basis(
    E_CAUSAL,
    Q8)

E_ORTH=[]
ORTH_FRAC=[]
PREORTH_COS=[]

for L in range(N_LAYERS):

    c=torch.dot(
        E_CAUSAL_GUARD[L],
        E_FISHER_GUARD[L])

    PREORTH_COS.append(float(c))

    r=(
        E_CAUSAL_GUARD[L]
        -c*E_FISHER_GUARD[L]
    )

    ORTH_FRAC.append(float(r.norm()))
    E_ORTH.append(norm(r))

E_ORTH=torch.stack(E_ORTH).contiguous()

print(
    f"       mean causal/Fisher cos="
    f"{np.mean(PREORTH_COS):+.6f}"
)

print(
    f"       mean Fisher-orthogonal component="
    f"{np.mean(ORTH_FRAC):.6f}"
)

# ======================================================================================================================
# [8/28] FROZEN ARM MASKS
# ======================================================================================================================

print("\n[8/28] Freezing terminal intervention arms...")

ARM_MULT={

"NORMAL":
    np.ones(N_LAYERS,dtype=np.float32),

"L19_OFF":
    np.asarray(
        [1.0]*19+[0.0],
        dtype=np.float32),

"L19_FLIP":
    np.asarray(
        [1.0]*19+[-1.0],
        dtype=np.float32),

"L18_19_OFF":
    np.asarray(
        [1.0]*18+[0.0,0.0],
        dtype=np.float32)
}

for arm,m in ARM_MULT.items():
    print(
        f"       {arm:<12} "
        f"L18={m[18]:+.1f} "
        f"L19={m[19]:+.1f}"
    )

# ======================================================================================================================
# [9/28] STEERING HOOKS
# ======================================================================================================================

def make_hooks(
    vectors,
    arm="NORMAL",
    sign=1.0,
    scale=.5):

    hs=[]
    mult=ARM_MULT[arm]

    for L in range(N_LAYERS):

        def mk(li):

            def hk(module,args,out):

                old=(
                    out[0]
                    if isinstance(out,tuple)
                    else out
                )

                rest=(
                    out[1:]
                    if isinstance(out,tuple)
                    else None
                )

                B=old.shape[0]

                direction=(
                    float(sign)
                    *vectors[li][None,:]
                    .expand(B,-1)
                ).float().contiguous()

                dose=torch.full(
                    (B,),
                    float(RHO[li])
                    *float(scale)
                    *float(mult[li]),
                    device=DEVICE,
                    dtype=torch.float32)

                new=seasc_ext.seasc_batch(
                    old,
                    direction,
                    dose)

                return (
                    new
                    if rest is None
                    else (new,)+rest
                )

            return hk

        hs.append(
            layers[L].register_forward_hook(
                mk(L)))

    return hs

# ======================================================================================================================
# [10/28] FRESH XRAY SET
# ======================================================================================================================

print("\n[10/28] Preparing fresh mechanistic X-ray prompts...")

XRAY=[
"A brief power interruption and an automatic system restart form a causal sequence.",
"A blocked drain and accumulated water form a causal sequence.",
"A sharp temperature drop and surface frost form a causal sequence.",
"A broken seal and loss of pressure form a causal sequence.",
"A strong push and movement of an object form a causal sequence.",
"A software fault and an unexpected shutdown form a causal sequence.",
"A damaged pipe and escaping water form a causal sequence.",
"A sudden voltage spike and component failure form a causal sequence."
]

@torch.inference_mode()
def trajectory(
    prompt,
    arm="NORMAL",
    sign=1.0,
    scale=.5):

    e=tokenizer(
        chat(prompt),
        return_tensors="pt").to(DEVICE)

    hs=make_hooks(
        E_ORTH,
        arm=arm,
        sign=sign,
        scale=scale)

    try:
        o=model(
            **e,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True)

    finally:
        for h in hs:
            h.remove()

    pos=int(
        e["attention_mask"][0].sum()
    )-1

    x=torch.stack([
        o.hidden_states[L+1][0,pos]
        .float()
        .detach()
        for L in range(N_LAYERS)
    ])

    del e,o

    return x

@torch.inference_mode()
def baseline_trajectory(prompt):

    e=tokenizer(
        chat(prompt),
        return_tensors="pt").to(DEVICE)

    o=model(
        **e,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True)

    pos=int(
        e["attention_mask"][0].sum()
    )-1

    x=torch.stack([
        o.hidden_states[L+1][0,pos]
        .float()
        .detach()
        for L in range(N_LAYERS)
    ])

    del e,o

    return x

# ======================================================================================================================
# [11/28] CAPTURE X-RAY
# ======================================================================================================================

print("\n[11/28] Capturing baseline trajectories...")

XR_BASE=[]

for i,p in enumerate(XRAY,1):
    XR_BASE.append(
        baseline_trajectory(p))
    print(f"       {i}/8")

XR_BASE=torch.stack(XR_BASE)

XR_PLUS={}
XR_MINUS={}

for arm in ARM_MULT:

    print(
        f"\n       Capturing {arm}...")

    pp=[]
    mm=[]

    for i,p in enumerate(XRAY,1):

        pp.append(
            trajectory(
                p,
                arm,
                +1,
                PRIMARY_SCALE))

        mm.append(
            trajectory(
                p,
                arm,
                -1,
                PRIMARY_SCALE))

        print(f"       {i}/8")

    XR_PLUS[arm]=torch.stack(pp)
    XR_MINUS[arm]=torch.stack(mm)

# ======================================================================================================================
# [12/28] DOWNSTREAM GEOMETRY
# ======================================================================================================================

print("\n[12/28] Computing terminal-ablation geometry...")

def geometry(arm):

    rows=[]

    for L in range(N_LAYERS):

        b=XR_BASE[:,L]

        d=.5*(
            (XR_PLUS[arm][:,L]-b)
            -(XR_MINUS[arm][:,L]-b)
        )

        bn=b.norm(
            dim=-1).clamp_min(EPS)

        dn=d.norm(
            dim=-1).clamp_min(EPS)

        rel=dn/bn
        target=d@E_ORTH[L]
        fisher=d@E_FISHER[L]

        perp=torch.sqrt(
            torch.clamp(
                dn.square()
                -target.square(),
                min=0))

        cos=target/dn

        angle=torch.rad2deg(
            torch.acos(
                cos.clamp(-1,1)))

        rows.append({
            "arm":arm,
            "layer":L,
            "rel":float(rel.mean()),
            "target":float(target.mean()),
            "fisher":float(fisher.mean()),
            "perp":float(perp.mean()),
            "cos_target":float(cos.mean()),
            "angle":float(angle.mean()),
            "delta_norm":float(dn.mean())
        })

    return rows

GEOM={
    arm:geometry(arm)
    for arm in ARM_MULT
}

for L in range(N_LAYERS):

    s=f"       L{L:02d}"

    for arm in ARM_MULT:

        r=GEOM[arm][L]

        s+=(
            f" | {arm} "
            f"tar={r['target']:+.3f} "
            f"cos={r['cos_target']:+.3f}"
        )

    print(s)

# ======================================================================================================================
# [13/28] TERMINAL FINGERPRINT
# ======================================================================================================================

print("\n[13/28] Terminal fingerprint...")

TERMINAL={}

for arm in ARM_MULT:

    r18=GEOM[arm][18]
    r19=GEOM[arm][19]

    late=GEOM[arm][15:20]

    TERMINAL[arm]={
        "late_target":
            float(np.mean([
                x["target"]
                for x in late
            ])),
        "late_cos":
            float(np.mean([
                x["cos_target"]
                for x in late
            ])),
        "L18_target":r18["target"],
        "L19_target":r19["target"],
        "L19_cos":r19["cos_target"],
        "L19_angle":r19["angle"],
        "L19_rel":r19["rel"]
    }

    t=TERMINAL[arm]

    print(
        f"       {arm:<12} "
        f"late={t['late_target']:+.6f} "
        f"L18={t['L18_target']:+.6f} "
        f"L19={t['L19_target']:+.6f} "
        f"cos={t['L19_cos']:+.6f} "
        f"angle={t['L19_angle']:.2f}°"
    )

# ======================================================================================================================
# [14/28] BEHAVIORAL DATASETS
# ======================================================================================================================

print("\n[14/28] Preparing fresh behavioral datasets...")

DEV=[
("A blocked filter caused the machine to overheat.",
 " The blocked filter is the cause.",
 " The overheating is the effect."),

("A loose electrical connection caused the light to flicker.",
 " The loose connection is the cause.",
 " The flickering is the effect."),

("A sharp impact caused the glass to fracture.",
 " The impact is the cause.",
 " The fracture is the effect."),

("A leaking valve caused the pressure to fall.",
 " The leaking valve is the cause.",
 " The pressure loss is the effect."),

("Heavy rain caused water to collect on the road.",
 " The heavy rain is the cause.",
 " The accumulated water is the effect."),

("A software error caused the application to stop.",
 " The software error is the cause.",
 " The application stopping is the effect."),

("Strong heating caused the material to expand.",
 " The heating is the cause.",
 " The expansion is the effect."),

("A failed bearing caused excessive vibration.",
 " The bearing failure is the cause.",
 " The excessive vibration is the effect.")
]

FINAL=[
("A damaged cable caused the device to lose power.",
 " The damaged cable is the cause.",
 " The loss of power is the effect."),

("A clogged pipe caused the tank to overflow.",
 " The clogged pipe is the cause.",
 " The overflow is the effect."),

("A sudden freeze caused the wet surface to become icy.",
 " The sudden freeze is the cause.",
 " The icy surface is the effect."),

("A strong force caused the object to accelerate.",
 " The force is the cause.",
 " The acceleration is the effect."),

("A failed pump caused the coolant flow to stop.",
 " The pump failure is the cause.",
 " The stopped coolant flow is the effect."),

("A short circuit caused the fuse to blow.",
 " The short circuit is the cause.",
 " The blown fuse is the effect."),

("A broken seal caused air to escape.",
 " The broken seal is the cause.",
 " The escaping air is the effect."),

("A collision caused the metal frame to bend.",
 " The collision is the cause.",
 " The bent frame is the effect.")
]

NEUTRAL=[
("A blue notebook rests beside a white sheet of paper.",
 " The notebook rests beside the paper."),

("A metal spoon lies next to a ceramic bowl.",
 " The spoon lies next to the bowl."),

("A small clock sits beside a closed book.",
 " The clock sits beside the book."),

("A glass bottle rests near a folded towel.",
 " The bottle rests near the towel."),

("A pencil lies beside an empty envelope.",
 " The pencil lies beside the envelope."),

("A wooden box sits next to a metal ruler.",
 " The box sits next to the ruler."),

("A calculator rests beside a plain folder.",
 " The calculator rests beside the folder."),

("A cup stands next to a small plate.",
 " The cup stands next to the plate.")
]

print("       DEV:",len(DEV))
print("       FINAL:",len(FINAL))
print("       NEUTRAL:",len(NEUTRAL))

# ======================================================================================================================
# [15/28] CONTINUATION LOG-PROB
# ======================================================================================================================

def continuation_logprob(
    context,
    continuation,
    arm=None,
    sign=1.0,
    scale=.5):

    ids,mask,plen=sequence(
        context,
        continuation)

    hs=[]

    if arm is not None:
        hs=make_hooks(
            E_ORTH,
            arm=arm,
            sign=sign,
            scale=scale)

    try:

        with torch.inference_mode():

            o=model(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                return_dict=True)

            z=o.logits[
                :,
                plen-1:-1,
                :
            ].float()

            tar=ids[:,plen:]

            lp=F.log_softmax(
                z,
                dim=-1)

            val=lp.gather(
                -1,
                tar.unsqueeze(-1)
            ).squeeze(-1).mean()

    finally:

        for h in hs:
            h.remove()

    return float(val)

# ======================================================================================================================
# [16/28] CAUSAL SCORE
# ======================================================================================================================

def evaluate_causal(data,arm):

    rows=[]

    for i,(ctx,pos,neg) in enumerate(data):

        bp=continuation_logprob(
            ctx,pos,None)

        bn=continuation_logprob(
            ctx,neg,None)

        pp=continuation_logprob(
            ctx,pos,arm,+1,PRIMARY_SCALE)

        pn=continuation_logprob(
            ctx,neg,arm,+1,PRIMARY_SCALE)

        mp=continuation_logprob(
            ctx,pos,arm,-1,PRIMARY_SCALE)

        mn=continuation_logprob(
            ctx,neg,arm,-1,PRIMARY_SCALE)

        base_margin=bp-bn
        plus_margin=pp-pn
        minus_margin=mp-mn

        # symmetric steering causal effect
        causal=.5*(
            (plus_margin-base_margin)
            -(minus_margin-base_margin)
        )

        rows.append({
            "item":i,
            "base_margin":base_margin,
            "plus_margin":plus_margin,
            "minus_margin":minus_margin,
            "causal":causal,
            "positive":causal>0
        })

    return rows

# ======================================================================================================================
# [17/28] NEUTRAL DISTURBANCE
# ======================================================================================================================

def evaluate_neutral(data,arm):

    rows=[]

    for i,(ctx,cont) in enumerate(data):

        b=continuation_logprob(
            ctx,cont,None)

        p=continuation_logprob(
            ctx,cont,arm,+1,PRIMARY_SCALE)

        m=continuation_logprob(
            ctx,cont,arm,-1,PRIMARY_SCALE)

        # symmetric absolute NLL/logprob disturbance proxy
        disturb=.5*(
            abs(p-b)
            +abs(m-b)
        )

        rows.append({
            "item":i,
            "base":b,
            "plus":p,
            "minus":m,
            "disturbance":disturb
        })

    return rows

# ======================================================================================================================
# [18/28] DEV — DIAGNOSTIC ONLY
# ======================================================================================================================

print("\n[18/28] Running fresh DEV diagnostic...")

DEV_RES={}

for arm in ARM_MULT:

    rr=evaluate_causal(
        DEV,
        arm)

    DEV_RES[arm]=rr

    vals=np.asarray([
        x["causal"]
        for x in rr])

    print(
        f"       {arm:<12} "
        f"C={vals.mean():+.6f} "
        f"P+={int((vals>0).sum())}/8"
    )

print("       No arm is selected from DEV.")

# ======================================================================================================================
# [19/28] FINAL — ALL PREREGISTERED ARMS
# ======================================================================================================================

print("\n[19/28] Opening FINAL for all preregistered arms...")

FINAL_RES={}

for arm in ARM_MULT:

    rr=evaluate_causal(
        FINAL,
        arm)

    FINAL_RES[arm]=rr

    vals=np.asarray([
        x["causal"]
        for x in rr])

    print(
        f"       {arm:<12} "
        f"C={vals.mean():+.6f} "
        f"P+={int((vals>0).sum())}/8"
    )

# ======================================================================================================================
# [20/28] NEUTRAL FINAL
# ======================================================================================================================

print("\n[20/28] Running held-out neutral disturbance...")

NEUTRAL_RES={}

for arm in ARM_MULT:

    rr=evaluate_neutral(
        NEUTRAL,
        arm)

    NEUTRAL_RES[arm]=rr

    vals=np.asarray([
        x["disturbance"]
        for x in rr])

    print(
        f"       {arm:<12} "
        f"|dNLL|={vals.mean():.6f}"
    )

# ======================================================================================================================
# [21/28] BOOTSTRAP
# ======================================================================================================================

print("\n[21/28] Bootstrap arm-vs-NORMAL behavioral effects...")

def bootstrap_diff(
    x,
    y,
    n=20000,
    seed=181):

    x=np.asarray(x,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)

    if len(x)!=len(y):
        raise ValueError(
            "Paired bootstrap requires equal lengths.")

    d=x-y

    rng=np.random.default_rng(seed)

    z=np.empty(n,dtype=np.float64)

    for i in range(n):
        ix=rng.integers(
            0,
            len(d),
            len(d))
        z[i]=d[ix].mean()

    return (
        float(d.mean()),
        float(np.quantile(z,.025)),
        float(np.quantile(z,.975))
    )

BOOT=[]

normal_c=[
    x["causal"]
    for x in FINAL_RES["NORMAL"]
]

normal_n=[
    x["disturbance"]
    for x in NEUTRAL_RES["NORMAL"]
]

for j,arm in enumerate(
    ["L19_OFF","L19_FLIP","L18_19_OFF"]):

    c=[
        x["causal"]
        for x in FINAL_RES[arm]
    ]

    n=[
        x["disturbance"]
        for x in NEUTRAL_RES[arm]
    ]

    cm,cl,ch=bootstrap_diff(
        c,
        normal_c,
        seed=SEED+j*100+1)

    nm,nl,nh=bootstrap_diff(
        n,
        normal_n,
        seed=SEED+j*100+2)

    BOOT.append({
        "arm":arm,
        "causal_delta":cm,
        "causal_ci_low":cl,
        "causal_ci_high":ch,
        "neutral_delta":nm,
        "neutral_ci_low":nl,
        "neutral_ci_high":nh
    })

    print(
        f"       {arm:<12} "
        f"ΔC={cm:+.6f} "
        f"[{cl:+.6f},{ch:+.6f}] "
        f"ΔNeutral={nm:+.6f} "
        f"[{nl:+.6f},{nh:+.6f}]"
    )

# ======================================================================================================================
# [22/28] TERMINAL GEOMETRY BOOTSTRAP
# ======================================================================================================================

print("\n[22/28] Bootstrap terminal target changes...")

def per_prompt_target(arm,L):

    b=XR_BASE[:,L]

    d=.5*(
        (XR_PLUS[arm][:,L]-b)
        -(XR_MINUS[arm][:,L]-b)
    )

    return (
        d@E_ORTH[L]
    ).detach().cpu().numpy()

TARGET_BOOT=[]

normal19=per_prompt_target(
    "NORMAL",
    19)

for j,arm in enumerate(
    ["L19_OFF","L19_FLIP","L18_19_OFF"]):

    x=per_prompt_target(
        arm,
        19)

    m,lo,hi=bootstrap_diff(
        x,
        normal19,
        seed=SEED+500+j)

    TARGET_BOOT.append({
        "arm":arm,
        "L19_target_delta":m,
        "ci_low":lo,
        "ci_high":hi
    })

    print(
        f"       {arm:<12} "
        f"ΔL19-target={m:+.6f} "
        f"[{lo:+.6f},{hi:+.6f}]"
    )

# ======================================================================================================================
# [23/28] MECHANISTIC COUPLING
# ======================================================================================================================

print("\n[23/28] Geometry ↔ behavior coupling...")

COUPLING=[]

normal_C=np.mean([
    x["causal"]
    for x in FINAL_RES["NORMAL"]
])

normal_T=TERMINAL["NORMAL"]["L19_target"]

for arm in ARM_MULT:

    C=np.mean([
        x["causal"]
        for x in FINAL_RES[arm]
    ])

    T=TERMINAL[arm]["L19_target"]

    COUPLING.append({
        "arm":arm,
        "L19_target":T,
        "causal_score":C,
        "delta_target_vs_normal":
            T-normal_T,
        "delta_causal_vs_normal":
            C-normal_C
    })

    print(
        f"       {arm:<12} "
        f"L19={T:+.6f} "
        f"C={C:+.6f} "
        f"ΔL19={T-normal_T:+.6f} "
        f"ΔC={C-normal_C:+.6f}"
    )

# ======================================================================================================================
# [24/28] PREREGISTERED DECISION
# ======================================================================================================================

print("\n[24/28] TEST181 diagnostic decision...")

def final_mean(arm):
    return float(np.mean([
        x["causal"]
        for x in FINAL_RES[arm]
    ]))

C_NORMAL=final_mean("NORMAL")
C_OFF=final_mean("L19_OFF")
C_FLIP=final_mean("L19_FLIP")
C_2OFF=final_mean("L18_19_OFF")

T_NORMAL=TERMINAL["NORMAL"]["L19_target"]
T_OFF=TERMINAL["L19_OFF"]["L19_target"]
T_FLIP=TERMINAL["L19_FLIP"]["L19_target"]
T_2OFF=TERMINAL["L18_19_OFF"]["L19_target"]

# Strongest support requires:
# 1) terminal manipulation changes L19 target in the intended direction
# 2) at least one terminal rescue arm improves behavioral causal score
# 3) behavioral bootstrap CI excludes zero for that rescue
#
# This avoids declaring causality from geometry alone.

SIG_RESCUE=[]

for b in BOOT:
    if (
        b["causal_delta"]>0
        and b["causal_ci_low"]>0
    ):
        SIG_RESCUE.append(
            b["arm"])

GEOM_CHANGED=(
    abs(T_OFF-T_NORMAL)>1e-6
    or abs(T_FLIP-T_NORMAL)>1e-6
    or abs(T_2OFF-T_NORMAL)>1e-6
)

if GEOM_CHANGED and len(SIG_RESCUE)>0:

    DECISION=(
        "TERMINAL_INTERVENTION_BEHAVIORAL_RESCUE_SUPPORTED"
    )

elif GEOM_CHANGED and max(
    C_OFF,
    C_FLIP,
    C_2OFF
)>C_NORMAL:

    DECISION=(
        "TERMINAL_INTERVENTION_POINT_ESTIMATE_RESCUE_ONLY"
    )

elif GEOM_CHANGED:

    DECISION=(
        "TERMINAL_GEOMETRY_CHANGED_WITHOUT_BEHAVIORAL_RESCUE"
    )

else:

    DECISION=(
        "TERMINAL_REVERSAL_CAUSAL_ROLE_NOT_RESOLVED"
    )

print("       NORMAL C :",f"{C_NORMAL:+.6f}")
print("       L19_OFF C:",f"{C_OFF:+.6f}")
print("       L19_FLIP C:",f"{C_FLIP:+.6f}")
print("       L18_19_OFF C:",f"{C_2OFF:+.6f}")

print("       NORMAL L19:",f"{T_NORMAL:+.6f}")
print("       OFF L19   :",f"{T_OFF:+.6f}")
print("       FLIP L19  :",f"{T_FLIP:+.6f}")
print("       2OFF L19  :",f"{T_2OFF:+.6f}")

print("       significant rescue arms:",SIG_RESCUE)
print("       DECISION:",DECISION)

# ======================================================================================================================
# [25/28] INTEGRITY
# ======================================================================================================================

print("\n[25/28] Integrity...")

INTEGRITY={
    "test180_mechanism_preserved":True,
    "axis":"CAUSE_EFFECT",
    "same_model":True,
    "same_seasc":True,
    "same_envelope":True,
    "same_p8_lineage":True,
    "same_causal_forge_lineage":True,
    "primary_scale":PRIMARY_SCALE,
    "arms_preregistered":True,
    "fresh_xray":True,
    "fresh_dev":True,
    "fresh_final":True,
    "fresh_neutral":True,
    "final_used_for_selection":False,
    "vector_search":False,
    "rank_search":False,
    "layer_search":False,
    "envelope_search":False,
    "controller":False,
    "weights_modified":False
}

for k,v in INTEGRITY.items():
    print(f"       {k}: {v}")

# ======================================================================================================================
# [26/28] SAVE
# ======================================================================================================================

print("\n[26/28] Saving...")

def savecsv(name,rows):

    if not rows:
        return

    keys=sorted(
        set().union(
            *[
                set(r.keys())
                for r in rows
            ]
        )
    )

    with open(
        ROOT/name,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        w=csv.DictWriter(
            f,
            fieldnames=keys)

        w.writeheader()
        w.writerows(rows)

ALL_GEOM=[]

for arm in ARM_MULT:
    ALL_GEOM+=GEOM[arm]

savecsv(
    "TERMINAL_GEOMETRY.csv",
    ALL_GEOM)

savecsv(
    "BOOTSTRAP_BEHAVIOR.csv",
    BOOT)

savecsv(
    "BOOTSTRAP_L19_TARGET.csv",
    TARGET_BOOT)

savecsv(
    "GEOMETRY_BEHAVIOR_COUPLING.csv",
    COUPLING)

for arm in ARM_MULT:

    savecsv(
        f"DEV_{arm}.csv",
        DEV_RES[arm])

    savecsv(
        f"FINAL_{arm}.csv",
        FINAL_RES[arm])

    savecsv(
        f"NEUTRAL_{arm}.csv",
        NEUTRAL_RES[arm])

np.savez_compressed(
    ROOT/"TEST181_TERMINAL_ABLATION.npz",

    rho=np.asarray(
        RHO,
        dtype=np.float32),

    E_fisher=
        E_FISHER.detach().cpu().numpy(),

    E_fisher_guard=
        E_FISHER_GUARD.detach().cpu().numpy(),

    E_causal=
        E_CAUSAL.detach().cpu().numpy(),

    E_orth=
        E_ORTH.detach().cpu().numpy(),

    xray_base=
        XR_BASE.detach().cpu().numpy(),

    **{
        f"{arm}_plus":
            XR_PLUS[arm]
            .detach()
            .cpu()
            .numpy()

        for arm in ARM_MULT
    },

    **{
        f"{arm}_minus":
            XR_MINUS[arm]
            .detach()
            .cpu()
            .numpy()

        for arm in ARM_MULT
    }
)

REPORT={
    "test":"TEST 181",
    "title":
        "TERMINAL REVERSAL CAUSAL ABLATION",

    "parent_test":
        "TEST180",

    "axis":
        "CAUSE_EFFECT",

    "seed":SEED,

    "model":
        MODEL_ID,

    "rss":
        RSS,

    "primary_scale":
        PRIMARY_SCALE,

    "fisher_bank":{
        "mean_cos":
            float(np.mean(FB_COS)),
        "mean_residual":
            float(np.mean(FB_RES))
    },

    "forge":{
        "mean_causal_fisher_cos":
            float(np.mean(PREORTH_COS)),
        "mean_orth_fraction":
            float(np.mean(ORTH_FRAC))
    },

    "terminal":
        TERMINAL,

    "bootstrap_behavior":
        BOOT,

    "bootstrap_target":
        TARGET_BOOT,

    "coupling":
        COUPLING,

    "significant_rescue_arms":
        SIG_RESCUE,

    "decision":
        DECISION,

    "integrity":
        INTEGRITY
}

with open(
    ROOT/"REPORT.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        REPORT,
        f,
        indent=2,
        ensure_ascii=False)

print("       Saved:",ROOT)

# ======================================================================================================================
# [27/28] SCIENTIFIC BOUNDARY
# ======================================================================================================================

print("\n[27/28] Scientific boundary...")

print(
    "       TEST181 intervenes on the terminal SEASC injection schedule; "
    "it does not modify or reselect the E synthetic vector."
)

print(
    "       L19_OFF and L19_FLIP test whether the terminal intervention "
    "contributes causally to the TEST180 terminal trajectory."
)

print(
    "       L18_19_OFF is a preregistered neighboring-layer control."
)

print(
    "       A behavioral rescue requires FINAL improvement, not merely "
    "a change in L19 geometry."
)

print(
    "       A significant rescue would support a causal role for the "
    "terminal intervention, but would not prove that terminal reversal "
    "is the unique cause of TEST178 failure."
)

print(
    "       Failure to rescue would indicate that the TEST180 L19 reversal "
    "is more likely a downstream symptom or an incomplete mechanistic marker."
)

# ======================================================================================================================
# [28/28] COMPLETE
# ======================================================================================================================

print("\n[28/28] COMPLETE")

print("="*154)
print("TEST 181 — TERMINAL REVERSAL CAUSAL ABLATION — SUMMARY")
print("="*154)

for arm in ARM_MULT:

    c=np.asarray([
        x["causal"]
        for x in FINAL_RES[arm]
    ])

    n=np.asarray([
        x["disturbance"]
        for x in NEUTRAL_RES[arm]
    ])

    t=TERMINAL[arm]

    print(
        f"{arm:<12} "
        f"C={c.mean():+.6f} "
        f"P+={int((c>0).sum())}/8 "
        f"Neutral={n.mean():.6f} "
        f"late_target={t['late_target']:+.6f} "
        f"L19_target={t['L19_target']:+.6f} "
        f"L19_cos={t['L19_cos']:+.6f} "
        f"L19_angle={t['L19_angle']:.2f}°"
    )

print("-"*154)

for b in BOOT:

    print(
        f"{b['arm']:<12} vs NORMAL "
        f"ΔC={b['causal_delta']:+.6f} "
        f"[{b['causal_ci_low']:+.6f},"
        f"{b['causal_ci_high']:+.6f}] "
        f"ΔNeutral={b['neutral_delta']:+.6f} "
        f"[{b['neutral_ci_low']:+.6f},"
        f"{b['neutral_ci_high']:+.6f}]"
    )

print("-"*154)
print("Significant behavioral rescue arms:",SIG_RESCUE)
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*154)
