# ======================================================================================================================
# TEST 179 — SUCCESS-vs-FAILURE DOWNSTREAM X-RAY
# AkbasCore 3.2 / SEASC
# TEST177 OPEN↔CLOSED (SUCCESS) vs TEST178 CAUSE↔EFFECT (FAILURE)
#
# QUESTION:
# Both TEST177 and TEST178 produced an almost Fisher-orthogonal causal vector.
# Why did OPEN↔CLOSED generalize strongly while CAUSE↔EFFECT did not?
#
# LOCKS:
# - Qwen/Qwen2.5-7B-Instruct BF16 / SDPA / A100
# - L0-L19, locked SEASC envelope
# - Same CUDA motor / same RSS
# - Reconstruct D and E independently with their original frozen recipes
# - P8 projection unchanged
# - X-RAY ONLY: no vector/rank/layer/envelope selection
# - Primary scale = 0.50
# - Fresh diagnostic prompts; TEST177/178 FINAL not reused
# - No controller / gate / brake / cap / boost / weight modification
# ======================================================================================================================
import os,sys,math,json,csv,random,shutil,subprocess,importlib.util
from pathlib import Path
if importlib.util.find_spec("ninja") is None:
    subprocess.check_call([sys.executable,"-m","pip","install","-q","ninja"])
import ninja
import numpy as np,torch,torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*154);print("TEST 179 — SUCCESS-vs-FAILURE DOWNSTREAM X-RAY");print("TEST177 OPEN↔CLOSED vs TEST178 CAUSE↔EFFECT");print("="*154)
os.environ["MAX_JOBS"]="2";os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available():raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda");SEED=179
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
MODEL_ID="Qwen/Qwen2.5-7B-Instruct";N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=.10,.30,.70,.20
FISHER_ALPHA,FISHER_EPS,EPS=.10,1e-8,1e-10
P8_COUNT=8;SVD_REL_TOL=1e-5;PRIMARY_SCALE=.50
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST179_SUCCESS_FAILURE_DOWNSTREAM_XRAY");ROOT.mkdir(parents=True,exist_ok=True)
def env(L):
    x=ZIRVE*math.exp(-SONUM*float(L))*(1+SONUM*float(L))+TABAN
    return x/(ZIRVE+TABAN)
ENVELOPE=np.asarray([env(L) for L in range(N_LAYERS)],dtype=np.float64);RHO=IVME*ENVELOPE
RSS=float(np.sqrt(np.sum(RHO**2)))
print("GPU:",torch.cuda.get_device_name(0));print("Model:",MODEL_ID);print(f"Locked RSS: {RSS:.9f}")

# ======================================================================================================================
# CUDA SEASC — UNCHANGED
# ======================================================================================================================
BUILD="/tmp/akbascore_test179";shutil.rmtree(BUILD,ignore_errors=True);os.makedirs(BUILD,exist_ok=True)
CPP=r"""#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
TORCH_CHECK(hidden.is_cuda()&&actuator.is_cuda()&&dose.is_cuda(),"CUDA tensors required");
TORCH_CHECK(hidden.dim()==3&&actuator.dim()==2&&dose.dim()==1,"bad dimensions");
TORCH_CHECK(hidden.size(0)==actuator.size(0)&&hidden.size(0)==dose.size(0)&&hidden.size(2)==actuator.size(1),"shape mismatch");
return seasc_batch_cuda(hidden,actuator,dose);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST179 SEASC");}"""
CUDA=r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename scalar_t>
__global__ void k(scalar_t* h,const float* a,const float* d,const int B,const int S,const int H){
int vec=blockIdx.x,b=vec/S;if(b>=B)return;extern __shared__ float sh[];
long long base=(long long)vec*H,ab=(long long)b*H;float ss=0.f;
for(int j=threadIdx.x;j<H;j+=blockDim.x){float x=(float)h[base+j];ss+=x*x;}sh[threadIdx.x]=ss;__syncthreads();
for(unsigned s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}
float n=sqrtf(fmaxf(sh[0],1e-20f)),scale=d[b]*n;__syncthreads();
for(int j=threadIdx.x;j<H;j+=blockDim.x){float x=(float)h[base+j];h[base+j]=(scalar_t)(x+scale*a[ab+j]);}}
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
auto out=hidden.contiguous().clone();auto act=actuator.to(hidden.device(),torch::kFloat32).contiguous();
auto d=dose.to(hidden.device(),torch::kFloat32).contiguous();int B=out.size(0),S=out.size(1),H=out.size(2),nv=B*S;
constexpr int T=256;size_t sb=T*sizeof(float);cudaStream_t stream=at::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"test179_seasc",[&]{
k<scalar_t><<<nv,T,sb,stream>>>(out.data_ptr<scalar_t>(),act.data_ptr<float>(),d.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}"""
print("\n[1/24] Compiling SEASC...")
seasc_ext=load_inline(name="akbascore_test179_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,
extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
print("       OK")

# ======================================================================================================================
# MODEL
# ======================================================================================================================
print("\n[2/24] Loading frozen model...")
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa",trust_remote_code=True)
model.eval()
for p in model.parameters():p.requires_grad_(False)
layers=model.model.layers;HIDDEN=model.config.hidden_size
def chat(x):return tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def norm(v):return v/v.norm(dim=-1,keepdim=True).clamp_min(EPS)
print("       hidden:",HIDDEN,"dtype:",next(model.parameters()).dtype)

@torch.inference_mode()
def capture(text):
    e=tokenizer(chat(text),return_tensors="pt").to(DEVICE)
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True);pos=int(e["attention_mask"][0].sum())-1
    hs=[o.hidden_states[L+1][0,pos].float().detach() for L in range(N_LAYERS)]
    del e,o;return hs

def sequence(context,continuation):
    p=tokenizer(chat(context),return_tensors="pt",add_special_tokens=False)
    c=tokenizer(continuation,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p["input_ids"],c["input_ids"]],1).to(DEVICE)
    return ids,torch.ones_like(ids),p["input_ids"].shape[1]

def activation_gradient(context,continuation,weighted=False,positive_score=False):
    ids,mask,plen=sequence(context,continuation);emb=model.get_input_embeddings()(ids).detach().requires_grad_(True)
    saved=[None]*N_LAYERS;hs=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(m,a,o):
                h=o[0] if isinstance(o,tuple) else o;h.retain_grad();saved[li]=h
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    model.zero_grad(set_to_none=True)
    try:
        o=model(inputs_embeds=emb,attention_mask=mask,use_cache=False,return_dict=True)
        z=o.logits[:,plen-1:-1,:].float();tar=ids[:,plen:];lp=F.log_softmax(z,-1)
        score=lp.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean()
        objective=score if positive_score else -score;objective.backward();gg=[]
        for L in range(N_LAYERS):
            g=saved[L].grad[0].float()
            if weighted:
                h=saved[L].detach()[0].float();g=(h.norm(dim=-1,keepdim=True)*g).sum(dim=0)
            else:g=g.sum(dim=0)
            gg.append(g.detach().clone())
        val=float(objective.detach())
    finally:
        for h in hs:h.remove()
        model.zero_grad(set_to_none=True)
    del ids,mask,emb,o,z,tar,lp,score,objective,saved
    return gg,val

# ======================================================================================================================
# FISHER/BANK BUILDER
# ======================================================================================================================
def build_axis(pairs,label):
    print(f"\n[3/24] Building {label} Fisher / Bank...")
    PB,NB=[[] for _ in range(N_LAYERS)],[[] for _ in range(N_LAYERS)]
    for i,(p,n) in enumerate(pairs,1):
        hp,hn=capture(p),capture(n)
        for L in range(N_LAYERS):PB[L].append(hp[L]);NB[L].append(hn[L])
        print(f"       {label} pair {i:02d}/{len(pairs)}")
    PB=[torch.stack(x).float() for x in PB];NB=[torch.stack(x).float() for x in NB]
    FI,BA,co,re=[],[],[],[]
    for L in range(N_LAYERS):
        Hp,Hn=PB[L],NB[L];mp,mn=Hp.mean(0),Hn.mean(0);md=mp-mn
        RR=torch.cat([Hp-mp,Hn-mn],0).float();nu=max(RR.shape[0]-2,1)
        _,S,Vh=torch.linalg.svd(RR,full_matrices=False);eig=S.square()/float(nu)
        pe=eig[eig>FISHER_EPS];es=torch.median(pe) if pe.numel() else torch.tensor(1.,device=DEVICE)
        lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS);V=Vh.T;pr=V.T@md
        f=norm(md/lam+V@((1/(eig+lam)-1/lam)*pr)).float()
        dirs=norm(Hp-Hn);dirs=dirs*torch.where((dirs@f)[:,None]<0,-torch.ones_like(dirs[:,:1]),torch.ones_like(dirs[:,:1]))
        b=norm(dirs.mean(0)).float();c=torch.dot(f,b).clamp(-1,1)
        if c<0:b,c=-b,-c
        FI.append(f);BA.append(b);co.append(float(c));re.append(float((b-c*f).norm()))
    print(f"       {label} mean cos(F,B)={np.mean(co):+.6f} residual={np.mean(re):.6f}")
    return torch.stack(FI).contiguous(),torch.stack(BA).contiguous(),co,re

D_PAIRS=[
("The wooden door is open.","The wooden door is closed."),("The glass window is open.","The glass window is closed."),
("The metal gate is open.","The metal gate is closed."),("The cardboard box is open.","The cardboard box is closed."),
("The desk drawer is open.","The desk drawer is closed."),("The storage cabinet is open.","The storage cabinet is closed."),
("The notebook is open.","The notebook is closed."),("The laptop is open.","The laptop is closed."),
("The suitcase is open.","The suitcase is closed."),("The envelope is open.","The envelope is closed."),
("The container lid is open.","The container lid is closed."),("The locker door is open.","The locker door is closed."),
("The hatch is open.","The hatch is closed."),("The entrance barrier is open.","The entrance barrier is closed."),
("The folding case is open.","The folding case is closed."),("The access panel is open.","The access panel is closed.")]

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
("The trigger is the cause of the response.","The response is the effect of the trigger.")]

D_FISHER,D_BANK,D_COS,D_RES=build_axis(D_PAIRS,"D OPEN↔CLOSED")
E_FISHER,E_BANK,E_COS,E_RES=build_axis(E_PAIRS,"E CAUSE↔EFFECT")

# ======================================================================================================================
# P8
# ======================================================================================================================
print("\n[4/24] Building shared frozen P8...")
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
    g,l=activation_gradient(ctx,cont);PG.append(g);print(f"       probe {i}/8")
Q8=[];RANKS=[]
for L in range(N_LAYERS):
    G=torch.stack([PG[i][L] for i in range(8)]).float();G=G[G.norm(dim=1)>EPS];G=norm(G)
    _,s,Vh=torch.linalg.svd(G,full_matrices=False);tol=max(float(s[0])*SVD_REL_TOL,EPS);rank=int((s>tol).sum())
    Q8.append(Vh[:rank].T.contiguous());RANKS.append(rank)
print("       ranks:",RANKS)

def project_basis(raw,bases):
    out=[]
    for L in range(N_LAYERS):
        u=raw[L].float();q=bases[L];v=u-(q@(q.T@u) if q.shape[1] else 0)
        if v.norm()<EPS:raise RuntimeError(f"Projection collapsed L{L}")
        out.append(norm(v))
    return torch.stack(out).contiguous()

D_FG=project_basis(D_FISHER,Q8);E_FG=project_basis(E_FISHER,Q8)

# ======================================================================================================================
# ORIGINAL TEST177/178 FORGE FAMILIES
# ======================================================================================================================
D_FORGE={
"A":[("A door is present, but its state is unspecified."," The door is open."," The door is closed."),
("A window is present, but its state is unspecified."," The window is open."," The window is closed."),
("A gate is present, but its state is unspecified."," The gate is open."," The gate is closed."),
("A box is present, but its state is unspecified."," The box is open."," The box is closed.")],
"B":[("A drawer is present. Its access state is unspecified."," The drawer allows access to its interior."," The drawer blocks access to its interior."),
("A cabinet is present. Its access state is unspecified."," The cabinet allows access to its interior."," The cabinet blocks access to its interior."),
("A suitcase is present. Its access state is unspecified."," The suitcase allows access to its interior."," The suitcase blocks access to its interior."),
("A locker is present. Its access state is unspecified."," The locker allows access to its interior."," The locker blocks access to its interior.")],
"C":[("Consider the physical state of a hatch."," The hatch is in an opened state."," The hatch is in a shut state."),
("Consider the physical state of an entrance barrier."," The barrier is in an opened state."," The barrier is in a shut state."),
("Consider the physical state of a folding case."," The case is in an opened state."," The case is in a shut state."),
("Consider the physical state of an access panel."," The panel is in an opened state."," The panel is in a shut state.")],
"D":[("A notebook has two possible physical states. No state is specified."," Its covers are separated so the interior pages are exposed."," Its covers are together so the interior pages are not exposed."),
("A laptop has two possible physical states. No state is specified."," Its display is raised away from the keyboard."," Its display is folded down against the keyboard."),
("An envelope has two possible physical states. No state is specified."," Its contents can be directly accessed through the opening."," Its contents are enclosed by the sealed flap."),
("A container has two possible lid states. No state is specified."," The lid leaves the container interior accessible."," The lid covers the container interior.")]}

E_FORGE={
"A":[("A spark and a fire are related causally. Which item is the initiating event?"," The spark is the cause."," The fire is the effect."),
("An impact and a crack are related causally. Which item is the initiating event?"," The impact is the cause."," The crack is the effect."),
("Rainfall and flooding are related causally. Which item is the initiating event?"," The rainfall is the cause."," The flooding is the effect."),
("Heat and melting are related causally. Which item is the initiating event?"," The heat is the cause."," The melting is the effect.")],
"B":[("A collision produces damage. Identify the causal role of the collision rather than its consequence."," The collision produces the damage."," The damage results from the collision."),
("Pressure produces deformation. Identify the causal role of the pressure rather than its consequence."," The pressure produces the deformation."," The deformation results from the pressure."),
("An infection produces fever. Identify the causal role of the infection rather than its consequence."," The infection produces the fever."," The fever results from the infection."),
("Vibration produces loosening. Identify the causal role of the vibration rather than its consequence."," The vibration produces the loosening."," The loosening results from the vibration.")],
"C":[("Consider the directional causal relation between frost and cracking."," Causally, frost precedes and generates the cracking."," Causally, cracking follows from the frost."),
("Consider the directional causal relation between blockage and overflow."," Causally, blockage precedes and generates the overflow."," Causally, overflow follows from the blockage."),
("Consider the directional causal relation between friction and heating."," Causally, friction precedes and generates the heating."," Causally, heating follows from the friction."),
("Consider the directional causal relation between force and motion."," Causally, force precedes and generates the motion."," Causally, motion follows from the force.")],
"D":[("An outage and a shutdown form a cause-effect pair. Select the upstream member."," The outage is upstream in the causal chain."," The shutdown is downstream in the causal chain."),
("A leak and pressure loss form a cause-effect pair. Select the upstream member."," The leak is upstream in the causal chain."," The pressure loss is downstream in the causal chain."),
("Acceleration and displacement form a cause-effect pair. Select the upstream member."," The acceleration is upstream in the causal chain."," The displacement is downstream in the causal chain."),
("A trigger and a response form a cause-effect pair. Select the upstream member."," The trigger is upstream in the causal chain."," The response is downstream in the causal chain.")]}

def forge_axis(FORGE,FISHER,FISHER_GUARD,label):
    print(f"\n[5/24] Reconstructing {label} causal vector...")
    FG={}
    for fn,items in FORGE.items():
        acc=[torch.zeros(HIDDEN,device=DEVICE,dtype=torch.float32) for _ in range(N_LAYERS)]
        for j,(ctx,pos,neg) in enumerate(items,1):
            gp,_=activation_gradient(ctx,pos,positive_score=True);gn,_=activation_gradient(ctx,neg,positive_score=True)
            for L in range(N_LAYERS):acc[L]+=gp[L]-gn[L]
            print(f"       {label} family {fn} {j}/4")
        v=torch.stack([norm(x) for x in acc]).contiguous()
        for L in range(N_LAYERS):
            if torch.dot(v[L],FISHER[L])<0:v[L]=-v[L]
        FG[fn]=v
    causal=torch.stack([norm(torch.stack([FG[k][L] for k in FG]).mean(0)) for L in range(N_LAYERS)]).contiguous()
    cg=project_basis(causal,Q8);out=[];frac=[]
    for L in range(N_LAYERS):
        r=cg[L]-torch.dot(cg[L],FISHER_GUARD[L])*FISHER_GUARD[L];frac.append(float(r.norm()));out.append(norm(r))
    out=torch.stack(out).contiguous()
    print(f"       {label} mean orthogonal component={np.mean(frac):.6f}")
    return out,FG,causal,frac

D_ORTH,D_FAMILIES,D_CAUSAL,D_FRAC=forge_axis(D_FORGE,D_FISHER,D_FG,"D")
E_ORTH,E_FAMILIES,E_CAUSAL,E_FRAC=forge_axis(E_FORGE,E_FISHER,E_FG,"E")

# ======================================================================================================================
# FRESH X-RAY PROMPTS
# ======================================================================================================================
print("\n[6/24] Preparing fresh X-ray prompts...")
D_XRAY=[
"A warehouse door is present, but its state is unspecified.",
"A laboratory window is present, but its state is unspecified.",
"A storage gate is present, but its state is unspecified.",
"A protective case is present, but its state is unspecified.",
"A cabinet door is present, but its state is unspecified.",
"A service panel is present, but its state is unspecified.",
"A transport box is present, but its state is unspecified.",
"An inspection hatch is present, but its state is unspecified."]

E_XRAY=[
"A sudden spark and a burned component form a causal sequence.",
"A strong impact and a fractured surface form a causal sequence.",
"Heavy rainfall and a flooded road form a causal sequence.",
"Intense heating and melted material form a causal sequence.",
"A collision and structural damage form a causal sequence.",
"High pressure and material deformation form a causal sequence.",
"An infection and elevated temperature form a causal sequence.",
"Persistent vibration and a loosened fastener form a causal sequence."]

# ======================================================================================================================
# LAYERWISE BASELINE / STEERED CAPTURE
# ======================================================================================================================
def make_hooks(vectors,sign,scale):
    hs=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(module,args,out):
                old=out[0] if isinstance(out,tuple) else out;rest=out[1:] if isinstance(out,tuple) else None
                B=old.shape[0];direction=(float(sign)*vectors[li][None,:].expand(B,-1)).float().contiguous()
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
    del e,o;return x

print("\n[7/24] Capturing D successful trajectories...")
D_BASE=[];D_PLUS=[];D_MINUS=[]
for i,p in enumerate(D_XRAY,1):
    D_BASE.append(trajectory(p));D_PLUS.append(trajectory(p,D_ORTH,+1,.5));D_MINUS.append(trajectory(p,D_ORTH,-1,.5))
    print(f"       D {i}/8")
D_BASE=torch.stack(D_BASE);D_PLUS=torch.stack(D_PLUS);D_MINUS=torch.stack(D_MINUS)

print("\n[8/24] Capturing E failed trajectories...")
E_BASE=[];E_PLUS=[];E_MINUS=[]
for i,p in enumerate(E_XRAY,1):
    E_BASE.append(trajectory(p));E_PLUS.append(trajectory(p,E_ORTH,+1,.5));E_MINUS.append(trajectory(p,E_ORTH,-1,.5))
    print(f"       E {i}/8")
E_BASE=torch.stack(E_BASE);E_PLUS=torch.stack(E_PLUS);E_MINUS=torch.stack(E_MINUS)

# ======================================================================================================================
# X-RAY METRICS
# ======================================================================================================================
def xray_metrics(label,B,P,M,V,F):
    rows=[]
    for L in range(N_LAYERS):
        b=B[:,L];dp=P[:,L]-b;dm=M[:,L]-b
        # symmetric causal displacement: (+ branch - - branch)/2
        d=.5*(dp-dm)
        bn=b.norm(dim=-1).clamp_min(EPS);dn=d.norm(dim=-1).clamp_min(EPS)
        rel=dn/bn
        target=(d@V[L])
        fisher=(d@F[L])
        perp=torch.sqrt(torch.clamp(dn.square()-target.square(),min=0))
        cos=target/dn
        angle=torch.rad2deg(torch.acos(cos.clamp(-1,1)))
        growth=rel/(float(RHO[L])*PRIMARY_SCALE+EPS)
        rows.append({"axis":label,"layer":L,"rel":float(rel.mean()),"target":float(target.mean()),
        "fisher_proj":float(fisher.mean()),"perp":float(perp.mean()),"cos_target":float(cos.mean()),
        "angle_deg":float(angle.mean()),"growth":float(growth.mean()),"delta_norm":float(dn.mean())})
    return rows

print("\n[9/24] Computing downstream geometry...")
D_ROWS=xray_metrics("D_SUCCESS",D_BASE,D_PLUS,D_MINUS,D_ORTH,D_FISHER)
E_ROWS=xray_metrics("E_FAILURE",E_BASE,E_PLUS,E_MINUS,E_ORTH,E_FISHER)
for L in range(N_LAYERS):
    d,e=D_ROWS[L],E_ROWS[L]
    print(f"       L{L:02d} D rel={d['rel']:.4f} cos={d['cos_target']:+.4f} target={d['target']:+.4f} | E rel={e['rel']:.4f} cos={e['cos_target']:+.4f} target={e['target']:+.4f}")

# ======================================================================================================================
# TRAJECTORY ROTATION
# ======================================================================================================================
print("\n[10/24] Measuring injected-direction rotation...")
ROT=[]
for label,B,P,M,V in [("D_SUCCESS",D_BASE,D_PLUS,D_MINUS,D_ORTH),("E_FAILURE",E_BASE,E_PLUS,E_MINUS,E_ORTH)]:
    for L in range(N_LAYERS):
        d=.5*((P[:,L]-B[:,L])-(M[:,L]-B[:,L]));dn=norm(d)
        c=(dn@V[L]).clamp(-1,1);a=torch.rad2deg(torch.acos(c))
        ROT.append({"axis":label,"layer":L,"mean_cos":float(c.mean()),"mean_angle":float(a.mean())})
print(f"       D L19 angle={ROT[19]['mean_angle']:.3f}°")
print(f"       E L19 angle={ROT[39]['mean_angle']:.3f}°")

# ======================================================================================================================
# CROSS-LAYER ALIGNMENT WITH FINAL SOURCE VECTOR
# ======================================================================================================================
print("\n[11/24] Measuring source-direction survival...")
SURV=[]
for label,B,P,M,V in [("D_SUCCESS",D_BASE,D_PLUS,D_MINUS,D_ORTH),("E_FAILURE",E_BASE,E_PLUS,E_MINUS,E_ORTH)]:
    src=V[0]
    for L in range(N_LAYERS):
        d=.5*((P[:,L]-B[:,L])-(M[:,L]-B[:,L]));c=norm(d)@src
        SURV.append({"axis":label,"layer":L,"cos_L00_source":float(c.mean())})
print("       complete")

# ======================================================================================================================
# TARGET/FISHER RATIO
# ======================================================================================================================
print("\n[12/24] Target-vs-Fisher projection...")
TF=[]
for d,e in zip(D_ROWS,E_ROWS):
    TF.append({"layer":d["layer"],"D_target":d["target"],"D_fisher":d["fisher_proj"],
    "E_target":e["target"],"E_fisher":e["fisher_proj"],
    "D_target_abs_ratio":abs(d["target"])/(abs(d["fisher_proj"])+EPS),
    "E_target_abs_ratio":abs(e["target"])/(abs(e["fisher_proj"])+EPS)})
print("       complete")

# ======================================================================================================================
# FAMILY COHERENCE
# ======================================================================================================================
print("\n[13/24] Comparing family coherence...")
FAM=[]
for label,FG in [("D_SUCCESS",D_FAMILIES),("E_FAILURE",E_FAMILIES)]:
    ks=list(FG)
    for a in range(len(ks)):
        for b in range(a+1,len(ks)):
            vals=[float(torch.dot(FG[ks[a]][L],FG[ks[b]][L])) for L in range(N_LAYERS)]
            FAM.append({"axis":label,"pair":ks[a]+"-"+ks[b],"mean_cos":float(np.mean(vals))})
            print(f"       {label} {ks[a]}-{ks[b]}={np.mean(vals):+.6f}")

# ======================================================================================================================
# FAMILY DISPERSION BY LAYER
# ======================================================================================================================
print("\n[14/24] Family dispersion by layer...")
FD=[]
for label,FG in [("D_SUCCESS",D_FAMILIES),("E_FAILURE",E_FAMILIES)]:
    for L in range(N_LAYERS):
        X=torch.stack([FG[k][L] for k in FG]);m=norm(X.mean(0))
        cs=X@m
        FD.append({"axis":label,"layer":L,"consensus_cos":float(cs.mean()),"consensus_min":float(cs.min()),"consensus_sd":float(cs.std())})
print("       complete")

# ======================================================================================================================
# DOWNSTREAM EFFECTIVE RANK / PCA
# ======================================================================================================================
print("\n[15/24] PCA / effective-rank analysis...")
PCA=[]
for label,B,P,M in [("D_SUCCESS",D_BASE,D_PLUS,D_MINUS),("E_FAILURE",E_BASE,E_PLUS,E_MINUS)]:
    for L in range(N_LAYERS):
        X=.5*((P[:,L]-B[:,L])-(M[:,L]-B[:,L]));X=X-X.mean(0,keepdim=True)
        _,s,_=torch.linalg.svd(X,full_matrices=False);var=s.square();frac=var/var.sum().clamp_min(EPS)
        er=float(torch.exp(-(frac*torch.log(frac.clamp_min(EPS))).sum()))
        PCA.append({"axis":label,"layer":L,"pc1":float(frac[0]),"pc12":float(frac[:2].sum()),"effective_rank":er})
print("       complete")

# ======================================================================================================================
# SUCCESS/FAILURE CONTRAST
# ======================================================================================================================
print("\n[16/24] Success/failure contrast...")
CONTRAST=[]
for L in range(N_LAYERS):
    d,e=D_ROWS[L],E_ROWS[L]
    CONTRAST.append({"layer":L,"D_minus_E_rel":d["rel"]-e["rel"],
    "D_minus_E_target":d["target"]-e["target"],
    "D_minus_E_cos":d["cos_target"]-e["cos_target"],
    "D_minus_E_perp":d["perp"]-e["perp"],
    "growth_ratio_D_over_E":d["growth"]/(e["growth"]+EPS)})
    print(f"       L{L:02d} Δrel={d['rel']-e['rel']:+.5f} Δtarget={d['target']-e['target']:+.5f} Δcos={d['cos_target']-e['cos_target']:+.5f}")

# ======================================================================================================================
# BOOTSTRAP LAYERWISE REL / TARGET / COS
# ======================================================================================================================
print("\n[17/24] Bootstrap D-vs-E layerwise...")
def per_prompt_metric(B,P,M,V,L):
    d=.5*((P[:,L]-B[:,L])-(M[:,L]-B[:,L]));dn=d.norm(dim=-1).clamp_min(EPS)
    rel=dn/B[:,L].norm(dim=-1).clamp_min(EPS);tar=d@V[L];co=tar/dn
    return rel.detach().cpu().numpy(),tar.detach().cpu().numpy(),co.detach().cpu().numpy()

def boot_unpaired(a,b,n=10000,seed=179):
    a=np.asarray(a);b=np.asarray(b);rng=np.random.default_rng(seed);z=np.empty(n)
    for i in range(n):
        aa=a[rng.integers(0,len(a),len(a))].mean();bb=b[rng.integers(0,len(b),len(b))].mean();z[i]=aa-bb
    return float(a.mean()-b.mean()),float(np.quantile(z,.025)),float(np.quantile(z,.975))

BSTR=[]
for L in range(N_LAYERS):
    dr,dt,dc=per_prompt_metric(D_BASE,D_PLUS,D_MINUS,D_ORTH,L)
    er,et,ec=per_prompt_metric(E_BASE,E_PLUS,E_MINUS,E_ORTH,L)
    for metric,a,b in [("REL",dr,er),("TARGET",dt,et),("COS",dc,ec)]:
        de,lo,hi=boot_unpaired(a,b,seed=SEED+L*31+len(metric))
        BSTR.append({"layer":L,"metric":metric,"D_minus_E":de,"ci_low":lo,"ci_high":hi})
print("       complete")

# ======================================================================================================================
# EARLY / MID / LATE ZONES
# ======================================================================================================================
print("\n[18/24] Zone summary...")
ZONES={"EARLY":range(0,5),"MID1":range(5,10),"MID2":range(10,15),"LATE":range(15,20)}
ZONE=[]
for zn,idx in ZONES.items():
    for label,rows in [("D_SUCCESS",D_ROWS),("E_FAILURE",E_ROWS)]:
        z={"zone":zn,"axis":label}
        for k in ["rel","target","perp","cos_target","growth"]:z[k]=float(np.mean([rows[L][k] for L in idx]))
        ZONE.append(z)
        print(f"       {zn:<5} {label:<9} rel={z['rel']:.4f} target={z['target']:+.4f} cos={z['cos_target']:+.4f} growth={z['growth']:.3f}")

# ======================================================================================================================
# PRIMARY MECHANISTIC FINGERPRINT
# ======================================================================================================================
print("\n[19/24] Primary mechanistic fingerprint...")
def mean_rows(rows,key):return float(np.mean([x[key] for x in rows]))
FINGERPRINT={
"D_mean_rel":mean_rows(D_ROWS,"rel"),"E_mean_rel":mean_rows(E_ROWS,"rel"),
"D_mean_target":mean_rows(D_ROWS,"target"),"E_mean_target":mean_rows(E_ROWS,"target"),
"D_mean_cos":mean_rows(D_ROWS,"cos_target"),"E_mean_cos":mean_rows(E_ROWS,"cos_target"),
"D_L19_rel":D_ROWS[19]["rel"],"E_L19_rel":E_ROWS[19]["rel"],
"D_L19_target":D_ROWS[19]["target"],"E_L19_target":E_ROWS[19]["target"],
"D_L19_cos":D_ROWS[19]["cos_target"],"E_L19_cos":E_ROWS[19]["cos_target"],
"D_orth_fraction":float(np.mean(D_FRAC)),"E_orth_fraction":float(np.mean(E_FRAC))}
for k,v in FINGERPRINT.items():print(f"       {k}: {v:+.6f}")

# ======================================================================================================================
# DIAGNOSTIC DECISION
# ======================================================================================================================
print("\n[20/24] TEST179 diagnostic decision...")
# Mechanistic, not efficacy selection.
lateD=np.mean([D_ROWS[L]["target"] for L in range(15,20)])
lateE=np.mean([E_ROWS[L]["target"] for L in range(15,20)])
cosD=np.mean([D_ROWS[L]["cos_target"] for L in range(15,20)])
cosE=np.mean([E_ROWS[L]["cos_target"] for L in range(15,20)])
relD=np.mean([D_ROWS[L]["rel"] for L in range(15,20)])
relE=np.mean([E_ROWS[L]["rel"] for L in range(15,20)])
if lateD>lateE and cosD>cosE:
    DECISION="SUCCESS_FAILURE_DOWNSTREAM_TARGET_RETENTION_DIFFERENCE_SUPPORTED"
elif abs(lateD-lateE)>EPS or abs(relD-relE)>EPS:
    DECISION="SUCCESS_FAILURE_DOWNSTREAM_TRAJECTORY_DIFFERENCE_OBSERVED"
else:
    DECISION="SUCCESS_FAILURE_DOWNSTREAM_DIFFERENCE_NOT_RESOLVED"
print("       DECISION:",DECISION)

# ======================================================================================================================
# INTERPRETATION FLAGS
# ======================================================================================================================
print("\n[21/24] Integrity / interpretation...")
FLAGS={
"xray_only":True,
"test177_success_axis":"OPEN_CLOSED",
"test178_failure_axis":"CAUSE_EFFECT",
"fresh_xray_prompts":True,
"test177_final_reused":False,
"test178_final_reused":False,
"vector_selection_performed":False,
"rank_search_performed":False,
"layer_search_performed":False,
"envelope_search_performed":False,
"controller":False,
"weights_modified":False}
for k,v in FLAGS.items():print(f"       {k}: {v}")

# ======================================================================================================================
# SAVE
# ======================================================================================================================
print("\n[22/24] Saving...")
def savecsv(name,rows):
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
savecsv("DOWNSTREAM_XRAY.csv",D_ROWS+E_ROWS);savecsv("ROTATION.csv",ROT);savecsv("SOURCE_SURVIVAL.csv",SURV)
savecsv("TARGET_FISHER.csv",TF);savecsv("FAMILY_COHERENCE.csv",FAM);savecsv("FAMILY_DISPERSION.csv",FD)
savecsv("PCA.csv",PCA);savecsv("CONTRAST.csv",CONTRAST);savecsv("BOOTSTRAP.csv",BSTR);savecsv("ZONE_SUMMARY.csv",ZONE)
np.savez_compressed(ROOT/"TEST179_SUCCESS_FAILURE_DOWNSTREAM_XRAY.npz",
d_orth=D_ORTH.detach().cpu().numpy(),e_orth=E_ORTH.detach().cpu().numpy(),
d_base=D_BASE.detach().cpu().numpy(),d_plus=D_PLUS.detach().cpu().numpy(),d_minus=D_MINUS.detach().cpu().numpy(),
e_base=E_BASE.detach().cpu().numpy(),e_plus=E_PLUS.detach().cpu().numpy(),e_minus=E_MINUS.detach().cpu().numpy(),
rho=np.asarray(RHO,dtype=np.float32))
REPORT={"test":"TEST 179","title":"SUCCESS-vs-FAILURE DOWNSTREAM X-RAY","model":MODEL_ID,"seed":SEED,
"primary_scale":PRIMARY_SCALE,"rss":RSS,"decision":DECISION,"fingerprint":FINGERPRINT,
"D_rows":D_ROWS,"E_rows":E_ROWS,"rotation":ROT,"source_survival":SURV,"target_fisher":TF,
"family_coherence":FAM,"family_dispersion":FD,"pca":PCA,"contrast":CONTRAST,
"bootstrap":BSTR,"zones":ZONE,"flags":FLAGS}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f:json.dump(REPORT,f,indent=2,ensure_ascii=False)
print("       Saved:",ROOT)

# ======================================================================================================================
# SCIENTIFIC BOUNDARY
# ======================================================================================================================
print("\n[23/24] Scientific boundary...")
print("       TEST179 selects no new vector or hyperparameter.")
print("       It compares downstream trajectories of one previously successful and one previously failed forge.")
print("       A trajectory difference is mechanistic evidence, not proof of a unique causal mechanism.")
print("       Similar Fisher-orthogonality alone is not treated as sufficient for behavioral success.")
print("       Fresh prompts prevent TEST177/178 FINAL evaluation reuse.")

# ======================================================================================================================
# COMPLETE
# ======================================================================================================================
print("\n[24/24] COMPLETE")
print("="*154);print("TEST 179 — SUCCESS-vs-FAILURE DOWNSTREAM X-RAY — SUMMARY");print("="*154)
print(f"D OPEN↔CLOSED orth fraction : {np.mean(D_FRAC):.6f}")
print(f"E CAUSE↔EFFECT orth fraction: {np.mean(E_FRAC):.6f}")
print(f"D mean rel/target/cos       : {FINGERPRINT['D_mean_rel']:.6f} / {FINGERPRINT['D_mean_target']:+.6f} / {FINGERPRINT['D_mean_cos']:+.6f}")
print(f"E mean rel/target/cos       : {FINGERPRINT['E_mean_rel']:.6f} / {FINGERPRINT['E_mean_target']:+.6f} / {FINGERPRINT['E_mean_cos']:+.6f}")
print(f"D L19 rel/target/cos        : {FINGERPRINT['D_L19_rel']:.6f} / {FINGERPRINT['D_L19_target']:+.6f} / {FINGERPRINT['D_L19_cos']:+.6f}")
print(f"E L19 rel/target/cos        : {FINGERPRINT['E_L19_rel']:.6f} / {FINGERPRINT['E_L19_target']:+.6f} / {FINGERPRINT['E_L19_cos']:+.6f}")
print("DECISION:",DECISION);print("Saved:",ROOT);print("="*154)
