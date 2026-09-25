# ======================================================================================================================
# TEST 180 — FIVE-AXIS FROZEN DOWNSTREAM VALIDATION
# AkbasCore 3.2 / SEASC
#
# A = ABOVE↔BELOW       (successful lineage: TEST169)
# B = BEFORE↔AFTER      (successful lineage: TEST175)
# C = INSIDE↔OUTSIDE    (successful lineage: TEST176)
# D = OPEN↔CLOSED       (successful lineage: TEST177)
# E = CAUSE↔EFFECT      (failed replication: TEST178)
#
# PURPOSE
# Validate the TEST179 candidate mechanistic fingerprint:
# Do successful synthetic vectors A/B/C/D preserve/recover downstream target-directed
# transport in late layers while failed E loses/reverses it?
#
# IMPORTANT
# - X-RAY ONLY
# - no hyperparameter/vector/rank/layer/envelope selection
# - same Qwen2.5-7B-Instruct
# - same frozen SEASC L0-L19
# - same RSS
# - same P8 construction
# - same causal-gradient forge recipe
# - fresh X-ray prompts
# - no previous FINAL prompts used
# - primary scale fixed at 0.50
# ======================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util,gc
from pathlib import Path
if importlib.util.find_spec("ninja") is None:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*154)
print("TEST 180 — FIVE-AXIS FROZEN DOWNSTREAM VALIDATION")
print("A/B/C/D SUCCESS vs E FAILURE — TEST179 MECHANISTIC FINGERPRINT VALIDATION")
print("="*154)

os.environ["MAX_JOBS"]="2"
os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required.")

DEVICE=torch.device("cuda")
SEED=180
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS=.10,1e-8,1e-10
SVD_REL_TOL=1e-5
PRIMARY_SCALE=.50
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST180_FIVE_AXIS_FROZEN_DOWNSTREAM_VALIDATION")
ROOT.mkdir(parents=True,exist_ok=True)

def env(L):
    x=ZIRVE*math.exp(-SONUM*float(L))*(1+SONUM*float(L))+TABAN
    return x/(ZIRVE+TABAN)

ENVELOPE=np.asarray([env(L) for L in range(N_LAYERS)],dtype=np.float64)
RHO=IVME*ENVELOPE
RSS=float(np.sqrt(np.sum(RHO**2)))

print("GPU:",torch.cuda.get_device_name(0))
print("Model:",MODEL_ID)
print("Axes: A ABOVE/BELOW | B BEFORE/AFTER | C INSIDE/OUTSIDE | D OPEN/CLOSED | E CAUSE/EFFECT")
print(f"Locked SEASC RSS: {RSS:.9f}")
print("Primary scale:",PRIMARY_SCALE)

# ======================================================================================================================
# [1/24] CUDA SEASC
# ======================================================================================================================
BUILD="/tmp/akbascore_test180"
shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)

CPP=r"""
#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
TORCH_CHECK(hidden.is_cuda()&&actuator.is_cuda()&&dose.is_cuda(),"CUDA tensors required");
TORCH_CHECK(hidden.dim()==3&&actuator.dim()==2&&dose.dim()==1,"bad dimensions");
TORCH_CHECK(hidden.size(0)==actuator.size(0)&&hidden.size(0)==dose.size(0)&&hidden.size(2)==actuator.size(1),"shape mismatch");
return seasc_batch_cuda(hidden,actuator,dose);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST180 SEASC");}
"""

CUDA=r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename scalar_t>
__global__ void k(scalar_t* h,const float* a,const float* d,const int B,const int S,const int H){
int vec=blockIdx.x,b=vec/S;if(b>=B)return;extern __shared__ float sh[];
long long base=(long long)vec*H,ab=(long long)b*H;float ss=0.f;
for(int j=threadIdx.x;j<H;j+=blockDim.x){float x=(float)h[base+j];ss+=x*x;}
sh[threadIdx.x]=ss;__syncthreads();
for(unsigned s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}
float n=sqrtf(fmaxf(sh[0],1e-20f)),scale=d[b]*n;__syncthreads();
for(int j=threadIdx.x;j<H;j+=blockDim.x){
float x=(float)h[base+j];h[base+j]=(scalar_t)(x+scale*a[ab+j]);}}
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
auto out=hidden.contiguous().clone();
auto act=actuator.to(hidden.device(),torch::kFloat32).contiguous();
auto d=dose.to(hidden.device(),torch::kFloat32).contiguous();
int B=out.size(0),S=out.size(1),H=out.size(2),nv=B*S;
constexpr int T=256;size_t sb=T*sizeof(float);
cudaStream_t stream=at::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"test180_seasc",[&]{
k<scalar_t><<<nv,T,sb,stream>>>(out.data_ptr<scalar_t>(),act.data_ptr<float>(),d.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}
"""

print("\n[1/24] Compiling frozen SEASC...")
seasc_ext=load_inline(
    name="akbascore_test180_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
    extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],
    with_cuda=True,build_directory=BUILD,verbose=False)
print("       OK")

# ======================================================================================================================
# [2/24] MODEL
# ======================================================================================================================
print("\n[2/24] Loading frozen model...")
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token

model=AutoModelForCausalLM.from_pretrained(
    MODEL_ID,dtype=torch.bfloat16,device_map={"":0},
    attn_implementation="sdpa",trust_remote_code=True)
model.eval()
for p in model.parameters():p.requires_grad_(False)

layers=model.model.layers
HIDDEN=model.config.hidden_size

def chat(x):
    return tokenizer.apply_chat_template(
        [{"role":"system","content":SYSTEM},{"role":"user","content":x}],
        tokenize=False,add_generation_prompt=True)

def norm(v):
    return v/v.norm(dim=-1,keepdim=True).clamp_min(EPS)

@torch.inference_mode()
def capture(text):
    e=tokenizer(chat(text),return_tensors="pt").to(DEVICE)
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True)
    pos=int(e["attention_mask"][0].sum())-1
    hs=[o.hidden_states[L+1][0,pos].float().detach() for L in range(N_LAYERS)]
    del e,o
    return hs

def sequence(context,continuation):
    p=tokenizer(chat(context),return_tensors="pt",add_special_tokens=False)
    c=tokenizer(continuation,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p["input_ids"],c["input_ids"]],1).to(DEVICE)
    return ids,torch.ones_like(ids),p["input_ids"].shape[1]

def activation_gradient(context,continuation,positive_score=False):
    ids,mask,plen=sequence(context,continuation)
    emb=model.get_input_embeddings()(ids).detach().requires_grad_(True)
    saved=[None]*N_LAYERS;hooks=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(m,a,o):
                h=o[0] if isinstance(o,tuple) else o
                h.retain_grad();saved[li]=h
            return hk
        hooks.append(layers[L].register_forward_hook(mk(L)))
    model.zero_grad(set_to_none=True)
    try:
        o=model(inputs_embeds=emb,attention_mask=mask,use_cache=False,return_dict=True)
        z=o.logits[:,plen-1:-1,:].float()
        tar=ids[:,plen:]
        lp=F.log_softmax(z,-1)
        score=lp.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean()
        objective=score if positive_score else -score
        objective.backward()
        gg=[]
        for L in range(N_LAYERS):
            g=saved[L].grad[0].float().sum(dim=0)
            gg.append(g.detach().clone())
    finally:
        for h in hooks:h.remove()
        model.zero_grad(set_to_none=True)
    return gg

print("       hidden:",HIDDEN,"dtype:",next(model.parameters()).dtype)

# ======================================================================================================================
# [3/24] AXIS DEFINITIONS
# ======================================================================================================================
print("\n[3/24] Loading five frozen semantic axis definitions...")

AXIS_PAIRS={

"A":[
("The bird is above the tree.","The bird is below the tree."),
("The lamp is above the table.","The lamp is below the table."),
("The cloud is above the mountain.","The cloud is below the mountain."),
("The sign is above the doorway.","The sign is below the doorway."),
("The shelf is above the desk.","The shelf is below the desk."),
("The bridge is above the river.","The bridge is below the river."),
("The picture is above the sofa.","The picture is below the sofa."),
("The ceiling is above the floor.","The ceiling is below the floor."),
("The airplane is above the city.","The airplane is below the city."),
("The balcony is above the entrance.","The balcony is below the entrance."),
("The clock is above the cabinet.","The clock is below the cabinet."),
("The branch is above the fence.","The branch is below the fence."),
("The roof is above the room.","The roof is below the room."),
("The kite is above the field.","The kite is below the field."),
("The helicopter is above the road.","The helicopter is below the road."),
("The moon is above the horizon.","The moon is below the horizon.")
],

"B":[
("The alarm happened before the meeting.","The alarm happened after the meeting."),
("The sunrise occurred before breakfast.","The sunrise occurred after breakfast."),
("The announcement came before the ceremony.","The announcement came after the ceremony."),
("The inspection occurred before the repair.","The inspection occurred after the repair."),
("The phone call happened before dinner.","The phone call happened after dinner."),
("The rain began before the match.","The rain began after the match."),
("The test occurred before the lesson.","The test occurred after the lesson."),
("The delivery arrived before lunch.","The delivery arrived after lunch."),
("The signal appeared before the response.","The signal appeared after the response."),
("The warning came before the shutdown.","The warning came after the shutdown."),
("The message arrived before the departure.","The message arrived after the departure."),
("The bell rang before the class.","The bell rang after the class."),
("The interview occurred before the decision.","The interview occurred after the decision."),
("The rehearsal happened before the performance.","The rehearsal happened after the performance."),
("The check occurred before the launch.","The check occurred after the launch."),
("The briefing happened before the operation.","The briefing happened after the operation.")
],

"C":[
("The coin is inside the box.","The coin is outside the box."),
("The book is inside the bag.","The book is outside the bag."),
("The key is inside the drawer.","The key is outside the drawer."),
("The ball is inside the basket.","The ball is outside the basket."),
("The letter is inside the envelope.","The letter is outside the envelope."),
("The tool is inside the cabinet.","The tool is outside the cabinet."),
("The bottle is inside the refrigerator.","The bottle is outside the refrigerator."),
("The toy is inside the container.","The toy is outside the container."),
("The document is inside the folder.","The document is outside the folder."),
("The clothes are inside the suitcase.","The clothes are outside the suitcase."),
("The vehicle is inside the garage.","The vehicle is outside the garage."),
("The animal is inside the cage.","The animal is outside the cage."),
("The person is inside the building.","The person is outside the building."),
("The package is inside the warehouse.","The package is outside the warehouse."),
("The pencil is inside the case.","The pencil is outside the case."),
("The food is inside the cupboard.","The food is outside the cupboard.")
],

"D":[
("The wooden door is open.","The wooden door is closed."),
("The glass window is open.","The glass window is closed."),
("The metal gate is open.","The metal gate is closed."),
("The cardboard box is open.","The cardboard box is closed."),
("The desk drawer is open.","The desk drawer is closed."),
("The storage cabinet is open.","The storage cabinet is closed."),
("The notebook is open.","The notebook is closed."),
("The laptop is open.","The laptop is closed."),
("The suitcase is open.","The suitcase is closed."),
("The envelope is open.","The envelope is closed."),
("The container lid is open.","The container lid is closed."),
("The locker door is open.","The locker door is closed."),
("The hatch is open.","The hatch is closed."),
("The entrance barrier is open.","The entrance barrier is closed."),
("The folding case is open.","The folding case is closed."),
("The access panel is open.","The access panel is closed.")
],

"E":[
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
]}

AXIS_NAME={
"A":"ABOVE↔BELOW",
"B":"BEFORE↔AFTER",
"C":"INSIDE↔OUTSIDE",
"D":"OPEN↔CLOSED",
"E":"CAUSE↔EFFECT"}

# ======================================================================================================================
# [4/24] FISHER/BANK
# ======================================================================================================================
def build_axis(pairs,label):
    PB,NB=[[] for _ in range(N_LAYERS)],[[] for _ in range(N_LAYERS)]
    for i,(p,n) in enumerate(pairs,1):
        hp,hn=capture(p),capture(n)
        for L in range(N_LAYERS):
            PB[L].append(hp[L]);NB[L].append(hn[L])
        print(f"       {label} pair {i:02d}/{len(pairs)}")
    PB=[torch.stack(x).float() for x in PB]
    NB=[torch.stack(x).float() for x in NB]
    FI=[];BA=[];co=[];re=[]
    for L in range(N_LAYERS):
        Hp,Hn=PB[L],NB[L]
        mp,mn=Hp.mean(0),Hn.mean(0);md=mp-mn
        RR=torch.cat([Hp-mp,Hn-mn],0).float()
        nu=max(RR.shape[0]-2,1)
        _,S,Vh=torch.linalg.svd(RR,full_matrices=False)
        eig=S.square()/float(nu)
        pe=eig[eig>FISHER_EPS]
        es=torch.median(pe) if pe.numel() else torch.tensor(1.,device=DEVICE)
        lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS)
        V=Vh.T;pr=V.T@md
        f=norm(md/lam+V@((1/(eig+lam)-1/lam)*pr)).float()
        dirs=norm(Hp-Hn)
        dirs=dirs*torch.where((dirs@f)[:,None]<0,-torch.ones_like(dirs[:,:1]),torch.ones_like(dirs[:,:1]))
        b=norm(dirs.mean(0)).float()
        c=torch.dot(f,b).clamp(-1,1)
        if c<0:b,c=-b,-c
        FI.append(f);BA.append(b);co.append(float(c));re.append(float((b-c*f).norm()))
    return torch.stack(FI).contiguous(),torch.stack(BA).contiguous(),co,re

print("\n[4/24] Reconstructing five independent Fisher/Bank axes...")
FISHER={};BANK={};FB_COS={};FB_RES={}
for k in "ABCDE":
    print(f"\n       AXIS {k} — {AXIS_NAME[k]}")
    FISHER[k],BANK[k],FB_COS[k],FB_RES[k]=build_axis(AXIS_PAIRS[k],k)
    print(f"       mean cos(F,B)={np.mean(FB_COS[k]):+.6f} residual={np.mean(FB_RES[k]):.6f}")

# ======================================================================================================================
# [5/24] SHARED P8
# ======================================================================================================================
print("\n[5/24] Building frozen P8...")
RETENTION_BUILD=[
("A ceramic cup rests beside a folded newspaper."," The scene contains ordinary stationary objects."),
("A metal ruler lies beside a closed notebook."," The scene contains ordinary stationary objects."),
("A wooden bowl sits near a plain cloth."," The scene contains ordinary stationary objects."),
("A glass jar rests beside a cardboard package."," The scene contains ordinary stationary objects."),
("A calculator lies next to a paper envelope."," The objects remain stationary."),
("A closed umbrella rests beside a travel bag."," The objects remain stationary."),
("A spoon lies beside an empty plate."," The objects remain stationary."),
("A pencil rests next to a blank sheet of paper."," The objects remain stationary.")]

PG=[]
for i,(ctx,cont) in enumerate(RETENTION_BUILD,1):
    PG.append(activation_gradient(ctx,cont))
    print(f"       probe {i}/8")

Q8=[];RANKS=[]
for L in range(N_LAYERS):
    G=torch.stack([PG[i][L] for i in range(8)]).float()
    G=G[G.norm(dim=1)>EPS];G=norm(G)
    _,s,Vh=torch.linalg.svd(G,full_matrices=False)
    tol=max(float(s[0])*SVD_REL_TOL,EPS)
    rank=int((s>tol).sum())
    Q8.append(Vh[:rank].T.contiguous());RANKS.append(rank)

print("       P8 ranks:",RANKS)

def project_basis(raw,bases):
    out=[]
    for L in range(N_LAYERS):
        u=raw[L].float();q=bases[L]
        v=u-(q@(q.T@u) if q.shape[1] else 0)
        if v.norm()<EPS:raise RuntimeError(f"Projection collapsed L{L}")
        out.append(norm(v))
    return torch.stack(out).contiguous()

FISHER_GUARD={k:project_basis(FISHER[k],Q8) for k in "ABCDE"}

# ======================================================================================================================
# [6/24] FROZEN CAUSAL FORGE DEFINITIONS
# ======================================================================================================================
print("\n[6/24] Loading frozen causal-gradient families...")

FORGE={}

FORGE["A"]={
"A":[
("An object and a reference point have a vertical relationship."," The object is above the reference point."," The object is below the reference point."),
("A marker has a vertical position relative to a line."," The marker is above the line."," The marker is below the line."),
("A point has a vertical position relative to a surface."," The point is above the surface."," The point is below the surface."),
("An item has a vertical location relative to another item."," The first item is above the second."," The first item is below the second.")],
"B":[
("Consider two vertically separated objects."," The first occupies the higher position."," The first occupies the lower position."),
("Consider a reference object and another object."," The other object is on the upper side."," The other object is on the lower side."),
("A target lies somewhere on the vertical axis relative to an anchor."," The target lies upward from the anchor."," The target lies downward from the anchor."),
("Two items differ only in vertical ordering."," The target is the upper item."," The target is the lower item.")],
"C":[
("A location is specified relative to a horizontal reference plane."," The location is on the upper side of the plane."," The location is on the lower side of the plane."),
("A symbol is positioned relative to a horizontal divider."," The symbol appears over the divider."," The symbol appears under the divider."),
("A point is displaced vertically from an origin."," The displacement is upward."," The displacement is downward."),
("A target differs from an anchor only along height."," The target has greater height."," The target has lesser height.")],
"D":[
("Select the positive vertical relation."," The relation points toward the upper direction."," The relation points toward the lower direction."),
("Describe the vertical ordering of two positions."," The first position is higher."," The first position is lower."),
("A target is offset from a reference along the vertical dimension."," The offset goes upward."," The offset goes downward."),
("A point occupies one side of a horizontal boundary."," It occupies the upper side."," It occupies the lower side.")]}

FORGE["B"]={
"A":[
("Two events have a temporal ordering."," The first event occurs before the second."," The first event occurs after the second."),
("An event is positioned relative to another event in time."," The event is earlier."," The event is later."),
("A target event has a temporal relation to a reference event."," The target precedes the reference."," The target follows the reference."),
("Two events differ only in temporal order."," The target happens first."," The target happens second.")],
"B":[
("Consider two points in a sequence of time."," The first is temporally earlier."," The first is temporally later."),
("An event is located on a timeline relative to an anchor event."," It lies before the anchor."," It lies after the anchor."),
("A target event has an ordering relation."," It comes prior to the reference."," It comes subsequent to the reference."),
("Two moments are ordered."," The target is the preceding moment."," The target is the following moment.")],
"C":[
("A temporal displacement is defined relative to a reference."," The displacement points toward an earlier time."," The displacement points toward a later time."),
("An occurrence has a position in chronological order."," It occurs previously."," It occurs subsequently."),
("A target is placed relative to an event boundary."," The target lies on the earlier side."," The target lies on the later side."),
("A sequence contains two ordered events."," The first precedes the second."," The first follows the second.")],
"D":[
("Select the earlier temporal relation."," The target occurs in advance of the reference."," The target occurs following the reference."),
("Describe the ordering of two events."," The first comes beforehand."," The first comes afterward."),
("A target is offset along the temporal dimension."," The offset goes toward the past."," The offset goes toward the future."),
("An event occupies one side of a temporal reference point."," It occupies the prior side."," It occupies the subsequent side.")]}

FORGE["C"]={
"A":[
("An object has a containment relation to a container."," The object is inside the container."," The object is outside the container."),
("A target has a location relative to a bounded region."," The target lies inside the region."," The target lies outside the region."),
("An item has a containment state relative to an enclosure."," The item is contained within it."," The item is external to it."),
("A point has a position relative to a boundary."," The point lies in the interior."," The point lies in the exterior.")],
"B":[
("Consider an object and an enclosing structure."," The object occupies the internal region."," The object occupies the external region."),
("A target is located relative to a container boundary."," It is within the boundary."," It is beyond the boundary."),
("An item differs only in containment status."," It is enclosed."," It is not enclosed."),
("A target has one of two spatial containment states."," It is internal."," It is external.")],
"C":[
("A location is defined relative to a closed region."," The location belongs to the interior."," The location belongs to the exterior."),
("A point is positioned relative to an enclosure."," The point is within the enclosure."," The point is beyond the enclosure."),
("An object is classified by containment."," The object is contained."," The object is uncontained."),
("A target is positioned relative to a bounded area."," The target is inward of the boundary."," The target is outward of the boundary.")],
"D":[
("Select the internal containment relation."," The target occupies the enclosed side."," The target occupies the non-enclosed side."),
("Describe the containment relation of an item."," The item lies within the container."," The item lies beyond the container."),
("A target is displaced relative to an enclosure boundary."," The target is toward the interior."," The target is toward the exterior."),
("A point occupies one of two regions separated by a boundary."," It occupies the inner region."," It occupies the outer region.")]}

FORGE["D"]={
"A":[
("A door is present, but its state is unspecified."," The door is open."," The door is closed."),
("A window is present, but its state is unspecified."," The window is open."," The window is closed."),
("A gate is present, but its state is unspecified."," The gate is open."," The gate is closed."),
("A box is present, but its state is unspecified."," The box is open."," The box is closed.")],
"B":[
("A drawer is present. Its access state is unspecified."," The drawer allows access to its interior."," The drawer blocks access to its interior."),
("A cabinet is present. Its access state is unspecified."," The cabinet allows access to its interior."," The cabinet blocks access to its interior."),
("A suitcase is present. Its access state is unspecified."," The suitcase allows access to its interior."," The suitcase blocks access to its interior."),
("A locker is present. Its access state is unspecified."," The locker allows access to its interior."," The locker blocks access to its interior.")],
"C":[
("Consider the physical state of a hatch."," The hatch is in an opened state."," The hatch is in a shut state."),
("Consider the physical state of an entrance barrier."," The barrier is in an opened state."," The barrier is in a shut state."),
("Consider the physical state of a folding case."," The case is in an opened state."," The case is in a shut state."),
("Consider the physical state of an access panel."," The panel is in an opened state."," The panel is in a shut state.")],
"D":[
("A notebook has two possible physical states. No state is specified."," Its covers are separated so the interior pages are exposed."," Its covers are together so the interior pages are not exposed."),
("A laptop has two possible physical states. No state is specified."," Its display is raised away from the keyboard."," Its display is folded down against the keyboard."),
("An envelope has two possible physical states. No state is specified."," Its contents can be directly accessed through the opening."," Its contents are enclosed by the sealed flap."),
("A container has two possible lid states. No state is specified."," The lid leaves the container interior accessible."," The lid covers the container interior.")]}

FORGE["E"]={
"A":[
("A spark and a fire are related causally. Which item is the initiating event?"," The spark is the cause."," The fire is the effect."),
("An impact and a crack are related causally. Which item is the initiating event?"," The impact is the cause."," The crack is the effect."),
("Rainfall and flooding are related causally. Which item is the initiating event?"," The rainfall is the cause."," The flooding is the effect."),
("Heat and melting are related causally. Which item is the initiating event?"," The heat is the cause."," The melting is the effect.")],
"B":[
("A collision produces damage. Identify the causal role of the collision rather than its consequence."," The collision produces the damage."," The damage results from the collision."),
("Pressure produces deformation. Identify the causal role of the pressure rather than its consequence."," The pressure produces the deformation."," The deformation results from the pressure."),
("An infection produces fever. Identify the causal role of the infection rather than its consequence."," The infection produces the fever."," The fever results from the infection."),
("Vibration produces loosening. Identify the causal role of the vibration rather than its consequence."," The vibration produces the loosening."," The loosening results from the vibration.")],
"C":[
("Consider the directional causal relation between frost and cracking."," Causally, frost precedes and generates the cracking."," Causally, cracking follows from the frost."),
("Consider the directional causal relation between blockage and overflow."," Causally, blockage precedes and generates the overflow."," Causally, overflow follows from the blockage."),
("Consider the directional causal relation between friction and heating."," Causally, friction precedes and generates the heating."," Causally, heating follows from the friction."),
("Consider the directional causal relation between force and motion."," Causally, force precedes and generates the motion."," Causally, motion follows from the force.")],
"D":[
("An outage and a shutdown form a cause-effect pair. Select the upstream member."," The outage is upstream in the causal chain."," The shutdown is downstream in the causal chain."),
("A leak and pressure loss form a cause-effect pair. Select the upstream member."," The leak is upstream in the causal chain."," The pressure loss is downstream in the causal chain."),
("Acceleration and displacement form a cause-effect pair. Select the upstream member."," The acceleration is upstream in the causal chain."," The displacement is downstream in the causal chain."),
("A trigger and a response form a cause-effect pair. Select the upstream member."," The trigger is upstream in the causal chain."," The response is downstream in the causal chain.")]}

# ======================================================================================================================
# [7/24] FORGE ALL FIVE AXES
# ======================================================================================================================
def forge_axis(key):
    FG={}
    for fn,items in FORGE[key].items():
        acc=[torch.zeros(HIDDEN,device=DEVICE,dtype=torch.float32) for _ in range(N_LAYERS)]
        for j,(ctx,pos,neg) in enumerate(items,1):
            gp=activation_gradient(ctx,pos,positive_score=True)
            gn=activation_gradient(ctx,neg,positive_score=True)
            for L in range(N_LAYERS):acc[L]+=gp[L]-gn[L]
            print(f"       {key} family {fn} {j}/4")
        v=torch.stack([norm(x) for x in acc]).contiguous()
        for L in range(N_LAYERS):
            if torch.dot(v[L],FISHER[key][L])<0:v[L]=-v[L]
        FG[fn]=v
    causal=torch.stack([
        norm(torch.stack([FG[f][L] for f in FG]).mean(0))
        for L in range(N_LAYERS)]).contiguous()
    cg=project_basis(causal,Q8)
    out=[];frac=[];fc=[]
    for L in range(N_LAYERS):
        fc.append(float(torch.dot(cg[L],FISHER_GUARD[key][L])))
        r=cg[L]-torch.dot(cg[L],FISHER_GUARD[key][L])*FISHER_GUARD[key][L]
        frac.append(float(r.norm()));out.append(norm(r))
    return torch.stack(out).contiguous(),FG,causal,frac,fc

print("\n[7/24] Reconstructing five frozen causal-gradient vectors...")
ORTH={};FAMILIES={};CAUSAL={};ORTH_FRAC={};PREORTH_COS={}
for k in "ABCDE":
    print(f"\n       AXIS {k} — {AXIS_NAME[k]}")
    ORTH[k],FAMILIES[k],CAUSAL[k],ORTH_FRAC[k],PREORTH_COS[k]=forge_axis(k)
    print(f"       mean causal/Fisher cos={np.mean(PREORTH_COS[k]):+.6f}")
    print(f"       mean Fisher-orthogonal component={np.mean(ORTH_FRAC[k]):.6f}")

# ======================================================================================================================
# [8/24] FRESH XRAY PROMPTS
# ======================================================================================================================
print("\n[8/24] Preparing fresh X-ray sets...")

XRAY={
"A":[
"A drone and a rooftop have an unspecified vertical relation.",
"A flag and a balcony have an unspecified vertical relation.",
"A balloon and a tree have an unspecified vertical relation.",
"A camera and a doorway have an unspecified vertical relation.",
"A light and a workbench have an unspecified vertical relation.",
"A bird and a fence have an unspecified vertical relation.",
"A sign and a vehicle have an unspecified vertical relation.",
"A marker and a horizontal line have an unspecified vertical relation."],

"B":[
"A warning and an inspection have an unspecified temporal ordering.",
"A rehearsal and an announcement have an unspecified temporal ordering.",
"A measurement and a repair have an unspecified temporal ordering.",
"A briefing and a departure have an unspecified temporal ordering.",
"A signal and a response have an unspecified temporal ordering.",
"A check and a shutdown have an unspecified temporal ordering.",
"A message and a meeting have an unspecified temporal ordering.",
"A test and a launch have an unspecified temporal ordering."],

"C":[
"A token and a container have an unspecified containment relation.",
"A tool and a storage case have an unspecified containment relation.",
"A document and a folder have an unspecified containment relation.",
"A package and a room have an unspecified containment relation.",
"A key and a cabinet have an unspecified containment relation.",
"A bottle and a cooler have an unspecified containment relation.",
"A device and an enclosure have an unspecified containment relation.",
"A marker and a bounded region have an unspecified containment relation."],

"D":[
"A warehouse door is present, but its state is unspecified.",
"A laboratory window is present, but its state is unspecified.",
"A storage gate is present, but its state is unspecified.",
"A protective case is present, but its state is unspecified.",
"A cabinet door is present, but its state is unspecified.",
"A service panel is present, but its state is unspecified.",
"A transport box is present, but its state is unspecified.",
"An inspection hatch is present, but its state is unspecified."],

"E":[
"A sudden spark and a burned component form a causal sequence.",
"A strong impact and a fractured surface form a causal sequence.",
"Heavy rainfall and a flooded road form a causal sequence.",
"Intense heating and melted material form a causal sequence.",
"A collision and structural damage form a causal sequence.",
"High pressure and material deformation form a causal sequence.",
"An infection and elevated temperature form a causal sequence.",
"Persistent vibration and a loosened fastener form a causal sequence."]}

# ======================================================================================================================
# [9/24] STEERING
# ======================================================================================================================
def make_hooks(vectors,sign,scale):
    hs=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(module,args,out):
                old=out[0] if isinstance(out,tuple) else out
                rest=out[1:] if isinstance(out,tuple) else None
                B=old.shape[0]
                direction=(float(sign)*vectors[li][None,:].expand(B,-1)).float().contiguous()
                dose=torch.full((B,),float(RHO[li])*float(scale),device=DEVICE,dtype=torch.float32)
                new=seasc_ext.seasc_batch(old,direction,dose)
                return new if rest is None else (new,)+rest
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    return hs

@torch.inference_mode()
def trajectory(prompt,v=None,sign=1,scale=.5):
    e=tokenizer(chat(prompt),return_tensors="pt").to(DEVICE)
    hs=make_hooks(v,sign,scale) if v is not None else []
    try:o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    pos=int(e["attention_mask"][0].sum())-1
    x=torch.stack([o.hidden_states[L+1][0,pos].float().detach() for L in range(N_LAYERS)])
    del e,o
    return x

print("\n[9/24] Capturing five-axis trajectories...")
BASE={};PLUS={};MINUS={}
for k in "ABCDE":
    BASE[k]=[];PLUS[k]=[];MINUS[k]=[]
    print(f"\n       AXIS {k} — {AXIS_NAME[k]}")
    for i,p in enumerate(XRAY[k],1):
        BASE[k].append(trajectory(p))
        PLUS[k].append(trajectory(p,ORTH[k],+1,PRIMARY_SCALE))
        MINUS[k].append(trajectory(p,ORTH[k],-1,PRIMARY_SCALE))
        print(f"       {i}/8")
    BASE[k]=torch.stack(BASE[k])
    PLUS[k]=torch.stack(PLUS[k])
    MINUS[k]=torch.stack(MINUS[k])

# ======================================================================================================================
# [10/24] METRICS
# ======================================================================================================================
def axis_metrics(k):
    rows=[]
    for L in range(N_LAYERS):
        b=BASE[k][:,L]
        d=.5*((PLUS[k][:,L]-b)-(MINUS[k][:,L]-b))
        bn=b.norm(dim=-1).clamp_min(EPS)
        dn=d.norm(dim=-1).clamp_min(EPS)
        rel=dn/bn
        target=d@ORTH[k][L]
        fisher=d@FISHER[k][L]
        perp=torch.sqrt(torch.clamp(dn.square()-target.square(),min=0))
        cos=target/dn
        angle=torch.rad2deg(torch.acos(cos.clamp(-1,1)))
        growth=rel/(float(RHO[L])*PRIMARY_SCALE+EPS)
        rows.append({
            "axis":k,"name":AXIS_NAME[k],"layer":L,
            "rel":float(rel.mean()),
            "target":float(target.mean()),
            "fisher":float(fisher.mean()),
            "perp":float(perp.mean()),
            "cos_target":float(cos.mean()),
            "angle":float(angle.mean()),
            "growth":float(growth.mean()),
            "delta_norm":float(dn.mean())})
    return rows

print("\n[10/24] Computing layerwise downstream geometry...")
ROWS={k:axis_metrics(k) for k in "ABCDE"}
for L in range(N_LAYERS):
    s=f"       L{L:02d}"
    for k in "ABCDE":
        r=ROWS[k][L]
        s+=f" | {k} rel={r['rel']:.3f} tar={r['target']:+.3f} cos={r['cos_target']:+.3f}"
    print(s)

# ======================================================================================================================
# [11/24] ZONE SUMMARY
# ======================================================================================================================
print("\n[11/24] Zone summaries...")
ZONES={"EARLY":range(0,5),"MID1":range(5,10),"MID2":range(10,15),"LATE":range(15,20)}
ZONE_ROWS=[]
for k in "ABCDE":
    for zn,idx in ZONES.items():
        z={"axis":k,"name":AXIS_NAME[k],"zone":zn}
        for metric in ["rel","target","cos_target","perp","growth"]:
            z[metric]=float(np.mean([ROWS[k][L][metric] for L in idx]))
        ZONE_ROWS.append(z)
        print(f"       {k} {zn:<5} rel={z['rel']:.4f} target={z['target']:+.4f} cos={z['cos_target']:+.4f} growth={z['growth']:.3f}")

# ======================================================================================================================
# [12/24] LATE TARGET RETENTION
# ======================================================================================================================
print("\n[12/24] Late-layer target-retention fingerprint...")
FINGERPRINT={}
for k in "ABCDE":
    late=ROWS[k][15:20]
    fp={
        "mean_rel":float(np.mean([r["rel"] for r in ROWS[k]])),
        "mean_target":float(np.mean([r["target"] for r in ROWS[k]])),
        "mean_cos":float(np.mean([r["cos_target"] for r in ROWS[k]])),
        "late_rel":float(np.mean([r["rel"] for r in late])),
        "late_target":float(np.mean([r["target"] for r in late])),
        "late_cos":float(np.mean([r["cos_target"] for r in late])),
        "L19_rel":ROWS[k][19]["rel"],
        "L19_target":ROWS[k][19]["target"],
        "L19_cos":ROWS[k][19]["cos_target"],
        "L19_angle":ROWS[k][19]["angle"],
        "orth_fraction":float(np.mean(ORTH_FRAC[k]))}
    FINGERPRINT[k]=fp
    print(f"       {k} {AXIS_NAME[k]:<15} late target={fp['late_target']:+.6f} late cos={fp['late_cos']:+.6f} L19 target={fp['L19_target']:+.6f} angle={fp['L19_angle']:.2f}°")

# ======================================================================================================================
# [13/24] TARGET SIGN SURVIVAL
# ======================================================================================================================
print("\n[13/24] Target-sign survival...")
SIGN_ROWS=[]
for k in "ABCDE":
    pos_layers=sum(ROWS[k][L]["target"]>0 for L in range(N_LAYERS))
    pos_late=sum(ROWS[k][L]["target"]>0 for L in range(15,20))
    sign_changes=0
    vals=[ROWS[k][L]["target"] for L in range(1,N_LAYERS)]
    for a,b in zip(vals[:-1],vals[1:]):
        if a*b<0:sign_changes+=1
    x={"axis":k,"positive_layers":pos_layers,"positive_late":pos_late,"sign_changes":sign_changes}
    SIGN_ROWS.append(x)
    print(f"       {k}: positive={pos_layers}/20 late={pos_late}/5 sign changes={sign_changes}")

# ======================================================================================================================
# [14/24] ROTATION
# ======================================================================================================================
print("\n[14/24] Rotation profile...")
ROT=[]
for k in "ABCDE":
    for L in range(N_LAYERS):
        b=BASE[k][:,L]
        d=.5*((PLUS[k][:,L]-b)-(MINUS[k][:,L]-b))
        c=(norm(d)@ORTH[k][L]).clamp(-1,1)
        a=torch.rad2deg(torch.acos(c))
        ROT.append({"axis":k,"layer":L,"cos":float(c.mean()),"angle":float(a.mean())})
    print(f"       {k} L19 angle={ROT[-1]['angle']:.3f}°")

# ======================================================================================================================
# [15/24] FAMILY COHERENCE
# ======================================================================================================================
print("\n[15/24] Family coherence...")
FAM_ROWS=[]
for k in "ABCDE":
    fs=list(FAMILIES[k])
    vals=[]
    for i in range(len(fs)):
        for j in range(i+1,len(fs)):
            cc=[float(torch.dot(FAMILIES[k][fs[i]][L],FAMILIES[k][fs[j]][L])) for L in range(N_LAYERS)]
            m=float(np.mean(cc));vals.append(m)
            FAM_ROWS.append({"axis":k,"pair":fs[i]+"-"+fs[j],"mean_cos":m})
    print(f"       {k} pairwise family mean={np.mean(vals):+.6f}")

# ======================================================================================================================
# [16/24] PCA / EFFECTIVE RANK
# ======================================================================================================================
print("\n[16/24] PCA / effective rank...")
PCA_ROWS=[]
for k in "ABCDE":
    for L in range(N_LAYERS):
        b=BASE[k][:,L]
        X=.5*((PLUS[k][:,L]-b)-(MINUS[k][:,L]-b))
        X=X-X.mean(0,keepdim=True)
        _,s,_=torch.linalg.svd(X,full_matrices=False)
        var=s.square();frac=var/var.sum().clamp_min(EPS)
        er=float(torch.exp(-(frac*torch.log(frac.clamp_min(EPS))).sum()))
        PCA_ROWS.append({"axis":k,"layer":L,"pc1":float(frac[0]),"pc12":float(frac[:2].sum()),"effective_rank":er})
print("       complete")

# ======================================================================================================================
# [17/24] BOOTSTRAP LATE TARGET
# ======================================================================================================================
print("\n[17/24] Bootstrap late target retention...")
def prompt_late_target(k):
    vals=[]
    for i in range(BASE[k].shape[0]):
        q=[]
        for L in range(15,20):
            b=BASE[k][i,L]
            d=.5*((PLUS[k][i,L]-b)-(MINUS[k][i,L]-b))
            q.append(float(torch.dot(d,ORTH[k][L])))
        vals.append(np.mean(q))
    return np.asarray(vals)

def boot_mean(x,n=20000,seed=180):
    rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):z[i]=x[rng.integers(0,len(x),len(x))].mean()
    return float(x.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))

BOOT=[]
for i,k in enumerate("ABCDE"):
    x=prompt_late_target(k)
    m,lo,hi=boot_mean(x,seed=SEED+i*101)
    BOOT.append({"axis":k,"late_target":m,"ci_low":lo,"ci_high":hi})
    print(f"       {k} late target={m:+.6f} [{lo:+.6f},{hi:+.6f}]")

# ======================================================================================================================
# [18/24] SUCCESS GROUP vs FAILURE
# ======================================================================================================================
print("\n[18/24] A/B/C/D success-group vs E...")
SUCCESS="ABCD"
SUCCESS_LATE=np.asarray([FINGERPRINT[k]["late_target"] for k in SUCCESS])
E_LATE=FINGERPRINT["E"]["late_target"]
SUCCESS_COS=np.asarray([FINGERPRINT[k]["late_cos"] for k in SUCCESS])
E_COS=FINGERPRINT["E"]["late_cos"]

print(f"       successful axes late target mean={SUCCESS_LATE.mean():+.6f}")
print(f"       failed E late target={E_LATE:+.6f}")
print(f"       successful axes late cos mean={SUCCESS_COS.mean():+.6f}")
print(f"       failed E late cos={E_COS:+.6f}")

# ======================================================================================================================
# [19/24] LEAVE-ONE-SUCCESS-OUT DESCRIPTIVE CHECK
# ======================================================================================================================
print("\n[19/24] Leave-one-success-out descriptive check...")
LOSO=[]
for held in SUCCESS:
    train=[k for k in SUCCESS if k!=held]
    floor=min(FINGERPRINT[k]["late_target"] for k in train)
    held_val=FINGERPRINT[held]["late_target"]
    e_val=FINGERPRINT["E"]["late_target"]
    row={"held":held,"train_floor":floor,"held_target":held_val,"E_target":e_val,
         "held_above_floor":held_val>=floor,"E_below_floor":e_val<floor}
    LOSO.append(row)
    print(f"       hold {held}: train floor={floor:+.6f} held={held_val:+.6f} E={e_val:+.6f}")

# ======================================================================================================================
# [20/24] PREREGISTERED DIAGNOSTIC
# ======================================================================================================================
print("\n[20/24] TEST180 validation decision...")

SUCCESS_POS=sum(FINGERPRINT[k]["late_target"]>0 for k in SUCCESS)
SUCCESS_L19_POS=sum(FINGERPRINT[k]["L19_target"]>0 for k in SUCCESS)
E_POS=FINGERPRINT["E"]["late_target"]>0
E_L19_POS=FINGERPRINT["E"]["L19_target"]>0

# This is intentionally strict:
# strongest replication requires all 4 successful axes positive in late mean and L19,
# while E breaks at least one of those two signatures.
if SUCCESS_POS==4 and SUCCESS_L19_POS==4 and (not E_POS or not E_L19_POS):
    DECISION="FOUR_SUCCESS_AXES_LATE_TARGET_SIGNATURE_REPLICATED__FAILED_AXIS_BREAKS_SIGNATURE"
elif SUCCESS_POS==4 and SUCCESS_L19_POS==4:
    DECISION="SUCCESS_AXES_SIGNATURE_REPLICATED__FAILED_AXIS_NOT_SEPARATED"
elif SUCCESS_POS>=3:
    DECISION="LATE_TARGET_SIGNATURE_PARTIAL_CROSS_CONCEPT_SUPPORT"
else:
    DECISION="LATE_TARGET_SIGNATURE_NOT_REPLICATED"

print("       successful late-positive:",SUCCESS_POS,"/4")
print("       successful L19-positive:",SUCCESS_L19_POS,"/4")
print("       E late-positive:",E_POS)
print("       E L19-positive:",E_L19_POS)
print("       DECISION:",DECISION)

# ======================================================================================================================
# [21/24] SCIENTIFIC FLAGS
# ======================================================================================================================
print("\n[21/24] Scientific flags...")
FLAGS={
"five_axes":True,
"successful_axes":["A_ABOVE_BELOW","B_BEFORE_AFTER","C_INSIDE_OUTSIDE","D_OPEN_CLOSED"],
"failed_axis":"E_CAUSE_EFFECT",
"xray_only":True,
"fresh_xray_prompts":True,
"previous_final_prompts_reused":False,
"final_used_for_selection":False,
"vector_selection":False,
"layer_selection":False,
"rank_selection":False,
"envelope_selection":False,
"runtime_controller":False,
"weights_modified":False,
"universal_rule_proven":False,
"test179_hypothesis_under_validation":True}
for k,v in FLAGS.items():print(f"       {k}: {v}")

# ======================================================================================================================
# [22/24] SAVE
# ======================================================================================================================
print("\n[22/24] Saving...")

def savecsv(name,rows):
    if not rows:return
    keys=sorted(set().union(*[set(r.keys()) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

ALL_ROWS=[]
for k in "ABCDE":ALL_ROWS+=ROWS[k]

savecsv("FIVE_AXIS_XRAY.csv",ALL_ROWS)
savecsv("ZONE_SUMMARY.csv",ZONE_ROWS)
savecsv("TARGET_SIGN_SURVIVAL.csv",SIGN_ROWS)
savecsv("ROTATION.csv",ROT)
savecsv("FAMILY_COHERENCE.csv",FAM_ROWS)
savecsv("PCA.csv",PCA_ROWS)
savecsv("BOOTSTRAP_LATE_TARGET.csv",BOOT)
savecsv("LOSO.csv",LOSO)

np.savez_compressed(
    ROOT/"TEST180_FIVE_AXIS_XRAY.npz",
    rho=np.asarray(RHO,dtype=np.float32),
    **{f"{k}_orth":ORTH[k].detach().cpu().numpy() for k in "ABCDE"},
    **{f"{k}_base":BASE[k].detach().cpu().numpy() for k in "ABCDE"},
    **{f"{k}_plus":PLUS[k].detach().cpu().numpy() for k in "ABCDE"},
    **{f"{k}_minus":MINUS[k].detach().cpu().numpy() for k in "ABCDE"})

REPORT={
"test":"TEST 180",
"title":"FIVE-AXIS FROZEN DOWNSTREAM VALIDATION",
"model":MODEL_ID,
"seed":SEED,
"rss":RSS,
"primary_scale":PRIMARY_SCALE,
"axis_names":AXIS_NAME,
"fingerprint":FINGERPRINT,
"zones":ZONE_ROWS,
"sign_survival":SIGN_ROWS,
"rotation":ROT,
"family_coherence":FAM_ROWS,
"pca":PCA_ROWS,
"bootstrap_late_target":BOOT,
"loso":LOSO,
"decision":DECISION,
"flags":FLAGS}

with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:
    json.dump(REPORT,f,indent=2,ensure_ascii=False)

print("       Saved:",ROOT)

# ======================================================================================================================
# [23/24] SCIENTIFIC BOUNDARY
# ======================================================================================================================
print("\n[23/24] Scientific boundary...")
print("       TEST180 performs no new vector or hyperparameter selection.")
print("       A/B/C/D are previously successful forge lineages; E is the previously failed fifth-axis lineage.")
print("       The test asks whether the TEST179 late target-retention observation generalizes across successful axes.")
print("       Similar source Fisher-orthogonality is not assumed sufficient for downstream behavioral success.")
print("       A repeated late-layer signature would be a mechanistic correlate, not proof of a universal law.")
print("       Failure of E to share the signature would not by itself prove that late target retention is the unique cause of failure.")

# ======================================================================================================================
# [24/24] COMPLETE
# ======================================================================================================================
print("\n[24/24] COMPLETE")
print("="*154)
print("TEST 180 — FIVE-AXIS FROZEN DOWNSTREAM VALIDATION — SUMMARY")
print("="*154)

for k in "ABCDE":
    f=FINGERPRINT[k]
    status="PRIOR-SUCCESS" if k in "ABCD" else "PRIOR-FAILURE"
    print(
        f"{k} {AXIS_NAME[k]:<15} {status:<13} "
        f"orth={f['orth_fraction']:.6f} "
        f"late_rel={f['late_rel']:.6f} "
        f"late_target={f['late_target']:+.6f} "
        f"late_cos={f['late_cos']:+.6f} "
        f"L19_target={f['L19_target']:+.6f} "
        f"L19_angle={f['L19_angle']:.2f}°")

print("-"*154)
print("Successful axes late-positive :",SUCCESS_POS,"/4")
print("Successful axes L19-positive  :",SUCCESS_L19_POS,"/4")
print("Failed E late-positive        :",E_POS)
print("Failed E L19-positive         :",E_L19_POS)
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*154)
