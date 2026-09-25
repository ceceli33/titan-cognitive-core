# TEST 168 — GUARDED SYNTHETIC VECTOR FORGE
# AkbasCore 3.2 / SEASC — single-cell compact implementation
# TEST167 working code retained. New variable: guarded Fisher+Bank-residual synthetic beta synthesis.
# SEARCH/FORGE/RETENTION basis is frozen before FINAL. FINAL never selects beta, guard, rank or thresholds.

import os,sys,math,json,csv,random,shutil,subprocess,gc
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*154); print("TEST 168 — GUARDED SYNTHETIC VECTOR FORGE"); print("RETENTION-NULLSPACE SYNTHESIS OF FISHER + BANK RESIDUAL"); print("="*154)
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers>=4.45.0","accelerate>=0.34.0","ninja","pybind11"])
os.environ["MAX_JOBS"]="2"; os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda"); SEED=168
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"; N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=0.10,0.30,0.70,0.20
FISHER_ALPHA,FISHER_EPS,EPS=0.10,1e-8,1e-10
PRIMARY_PROBES=8; SVD_REL_TOL=1e-5
COARSE_BETAS=np.arange(-2.0,2.0001,0.25).round(6).tolist()
FINE_STEP=0.10; FINE_RADIUS=0.25; MICRO_DELTA=0.05
FORGE_SCALE=0.50; SCALES=[0.125,0.25,0.50,0.75,1.00]; PRIMARY_SCALE=0.50
CAUSAL_FLOOR=0.90; KL_TOL=1.20; RET_TOL=1.20; DPERP_TOL=1.20
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST168_GUARDED_SYNTHETIC_VECTOR_FORGE"); ROOT.mkdir(parents=True,exist_ok=True)
print("GPU:",torch.cuda.get_device_name(0)); print("Model:",MODEL_ID); print("Layers: L0-L19 | guard=P8 | coarse beta:",COARSE_BETAS)

# =================================================================================================
# CUDA SEASC — identical working structure from TEST167
# =================================================================================================
BUILD="/tmp/akbascore_test168"; shutil.rmtree(BUILD,ignore_errors=True); os.makedirs(BUILD,exist_ok=True)
CPP=r"""#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
TORCH_CHECK(hidden.is_cuda()&&actuator.is_cuda()&&dose.is_cuda(),"CUDA tensors required");
TORCH_CHECK(hidden.dim()==3&&actuator.dim()==2&&dose.dim()==1,"bad dimensions");
TORCH_CHECK(hidden.size(0)==actuator.size(0)&&hidden.size(0)==dose.size(0)&&hidden.size(2)==actuator.size(1),"shape mismatch");
return seasc_batch_cuda(hidden,actuator,dose);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST168 SEASC");}"""
CUDA=r"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
template<typename scalar_t>
__global__ void k(scalar_t* h,const float* a,const float* d,const int B,const int S,const int H){
int vec=blockIdx.x,b=vec/S;if(b>=B)return;extern __shared__ float sh[];
long long base=(long long)vec*H,ab=(long long)b*H;float ss=0.f;
for(int j=threadIdx.x;j<H;j+=blockDim.x){float x=(float)h[base+j];ss+=x*x;} sh[threadIdx.x]=ss;__syncthreads();
for(unsigned s=blockDim.x/2;s>0;s>>=1){if(threadIdx.x<s)sh[threadIdx.x]+=sh[threadIdx.x+s];__syncthreads();}
float n=sqrtf(fmaxf(sh[0],1e-20f)),scale=d[b]*n;__syncthreads();
for(int j=threadIdx.x;j<H;j+=blockDim.x){float x=(float)h[base+j];h[base+j]=(scalar_t)(x+scale*a[ab+j]);}}
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
auto out=hidden.contiguous().clone();auto act=actuator.to(hidden.device(),torch::kFloat32).contiguous();auto d=dose.to(hidden.device(),torch::kFloat32).contiguous();
int B=out.size(0),S=out.size(1),H=out.size(2),nv=B*S;constexpr int T=256;size_t sb=T*sizeof(float);cudaStream_t stream=at::cuda::getCurrentCUDAStream();
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"test168_seasc",[&]{k<scalar_t><<<nv,T,sb,stream>>>(out.data_ptr<scalar_t>(),act.data_ptr<float>(),d.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}"""
print("\n[1/18] Compiling SEASC...")
seasc_ext=load_inline(name="akbascore_test168_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
print("       OK")

# =================================================================================================
# MODEL
# =================================================================================================
print("\n[2/18] Loading frozen model...")
tokenizer=AutoTokenizer.from_pretrained(MODEL_ID,trust_remote_code=True)
if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.bfloat16,device_map={"":0},attn_implementation="sdpa",trust_remote_code=True)
model.eval()
for p in model.parameters(): p.requires_grad_(False)
layers=model.model.layers; HIDDEN=model.config.hidden_size
if len(layers)<N_LAYERS: raise RuntimeError("Model has fewer than 20 layers.")
def env(L):
    x=ZIRVE*math.exp(-SONUM*float(L))*(1+SONUM*float(L))+TABAN
    return x/(ZIRVE+TABAN)
ENVELOPE=[env(L) for L in range(N_LAYERS)]
def chat(x): return tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":x}],tokenize=False,add_generation_prompt=True)
def norm(v): return v/v.norm(dim=-1,keepdim=True).clamp_min(EPS)
print("       hidden:",HIDDEN,"dtype:",next(model.parameters()).dtype)

# =================================================================================================
# SEARCH GEOMETRY — identical TEST167 Fisher / Bank / Residual construction
# =================================================================================================
SEARCH_PAIRS=[
("The lantern is above the stone arch.","The lantern is below the stone arch."),
("The kite is above the old tower.","The kite is below the old tower."),
("The flag is above the wooden gate.","The flag is below the wooden gate."),
("The bird is above the narrow bridge.","The bird is below the narrow bridge."),
("The sign is above the glass doorway.","The sign is below the glass doorway."),
("The lamp is above the metal shelf.","The lamp is below the metal shelf."),
("The clock is above the brown cabinet.","The clock is below the brown cabinet."),
("The balloon is above the green hill.","The balloon is below the green hill."),
("The camera is above the small desk.","The camera is below the small desk."),
("The painting is above the blue sofa.","The painting is below the blue sofa."),
("The cloud is above the distant ridge.","The cloud is below the distant ridge."),
("The marker is above the white board.","The marker is below the white board."),
("The drone is above the concrete wall.","The drone is below the concrete wall."),
("The shelf is above the storage box.","The shelf is below the storage box."),
("The light is above the front window.","The light is below the front window."),
("The symbol is above the horizontal line.","The symbol is below the horizontal line.")]

@torch.inference_mode()
def capture(text):
    e=tokenizer(chat(text),return_tensors="pt").to(DEVICE)
    o=model(**e,output_hidden_states=True,use_cache=False,return_dict=True)
    pos=int(e["attention_mask"][0].sum())-1
    hs=[o.hidden_states[L+1][0,pos].float().detach() for L in range(N_LAYERS)]
    del e,o
    return hs

print("\n[3/18] Building Fisher / Bank / Residual geometry...")
PB,NB=[[] for _ in range(N_LAYERS)],[[] for _ in range(N_LAYERS)]
for i,(p,n) in enumerate(SEARCH_PAIRS,1):
    hp,hn=capture(p),capture(n)
    for L in range(N_LAYERS): PB[L].append(hp[L]); NB[L].append(hn[L])
    print(f"       pair {i:02d}/16")
PB=[torch.stack(x).float() for x in PB]; NB=[torch.stack(x).float() for x in NB]
FISHER,BANK,RESIDUAL,COS,RESN=[],[],[],[],[]
for L in range(N_LAYERS):
    Hp,Hn=PB[L],NB[L]; mp,mn=Hp.mean(0),Hn.mean(0); md=mp-mn
    R=torch.cat([Hp-mp,Hn-mn],0).float(); nu=max(R.shape[0]-2,1)
    _,S,Vh=torch.linalg.svd(R,full_matrices=False); eig=S.square()/float(nu)
    pe=eig[eig>FISHER_EPS]; es=torch.median(pe) if pe.numel() else torch.tensor(1.,device=DEVICE)
    lam=(FISHER_ALPHA*es).clamp_min(FISHER_EPS); V=Vh.T; pr=V.T@md
    f=norm(md/lam+V@((1/(eig+lam)-1/lam)*pr)).float()
    dirs=norm(Hp-Hn); dirs=dirs*torch.where((dirs@f)[:,None]<0,-torch.ones_like(dirs[:,:1]),torch.ones_like(dirs[:,:1]))
    b=norm(dirs.mean(0)).float(); c=torch.dot(f,b).clamp(-1,1)
    if c<0: b,c=-b,-c
    r=b-c*f; rn=r.norm()
    if rn<EPS: raise RuntimeError(f"Degenerate residual L{L}")
    r=(r/rn).float(); FISHER.append(f); BANK.append(b); RESIDUAL.append(r); COS.append(float(c)); RESN.append(float(rn))
    print(f"       L{L:02d} cos(F,B)={float(c):+.6f} residual={float(rn):.6f}")
FISHER=torch.stack(FISHER).contiguous(); BANK=torch.stack(BANK).contiguous(); RESIDUAL=torch.stack(RESIDUAL).contiguous()
print(f"       mean cos={np.mean(COS):+.6f} | mean residual={np.mean(RESN):.6f}")

def beta_vector(beta):
    q=[]
    for L in range(N_LAYERS):
        ratio=RESN[L]/max(COS[L],1e-8)
        q.append(norm(FISHER[L]+float(beta)*ratio*RESIDUAL[L]).float())
    return torch.stack(q).contiguous()

# =================================================================================================
# RETENTION P8 — same TEST167 definition
# =================================================================================================
RETENTION=[
("A ceramic cup rests beside a folded newspaper."," The scene contains ordinary stationary objects."),
("A metal ruler lies beside a closed notebook."," The scene contains ordinary stationary objects."),
("A wooden bowl sits near a plain cloth."," The scene contains ordinary stationary objects."),
("A glass jar rests beside a cardboard package."," The scene contains ordinary stationary objects."),
("A calculator lies next to a paper envelope."," The objects remain stationary."),
("A closed umbrella rests beside a travel bag."," The objects remain stationary."),
("A spoon lies beside an empty plate."," The objects remain stationary."),
("A pencil rests next to a blank sheet of paper."," The objects remain stationary.")]

def sequence(context,continuation):
    p=tokenizer(chat(context),return_tensors="pt",add_special_tokens=False)
    c=tokenizer(continuation,return_tensors="pt",add_special_tokens=False)
    ids=torch.cat([p["input_ids"],c["input_ids"]],1).to(DEVICE); plen=p["input_ids"].shape[1]
    return ids,torch.ones_like(ids),plen

def activation_gradient(context,continuation):
    ids,mask,plen=sequence(context,continuation)
    embeds=model.get_input_embeddings()(ids).detach().requires_grad_(True)
    saved=[None]*N_LAYERS; handles=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(module,args,out):
                h=out[0] if isinstance(out,tuple) else out
                h.retain_grad(); saved[li]=h
            return hk
        handles.append(layers[L].register_forward_hook(mk(L)))
    model.zero_grad(set_to_none=True)
    try:
        out=model(inputs_embeds=embeds,attention_mask=mask,use_cache=False,return_dict=True)
        logits=out.logits.float(); target=ids[:,plen:]; pred=logits[:,plen-1:-1,:]
        lp=F.log_softmax(pred,-1); loss=-lp.gather(-1,target.unsqueeze(-1)).squeeze(-1).mean()
        loss.backward(); grads=[]
        for L in range(N_LAYERS):
            if saved[L] is None or saved[L].grad is None: raise RuntimeError(f"No activation gradient at L{L}")
            g=saved[L].grad[0].float().sum(dim=0).detach()
            if not torch.isfinite(g).all(): raise RuntimeError(f"Nonfinite retention gradient L{L}")
            grads.append(g)
        lv=float(loss.detach())
    finally:
        for h in handles:h.remove()
        model.zero_grad(set_to_none=True)
    del ids,mask,embeds,out,logits,target,pred,lp,loss,saved
    return grads,lv

print("\n[4/18] Computing P8 retention-sensitive activation gradients...")
PROBE_GRADS=[]; RET_LOSSES=[]
for i,(ctx,cont) in enumerate(RETENTION,1):
    g,l=activation_gradient(ctx,cont); PROBE_GRADS.append(g); RET_LOSSES.append(l)
    print(f"       probe {i}/8 | NLL={l:.6f} | mean ||g||={np.mean([float(x.norm()) for x in g]):.6e}")
    torch.cuda.empty_cache()

def build_basis():
    bases=[]; ranks=[]; singular=[]
    for L in range(N_LAYERS):
        G=torch.stack([PROBE_GRADS[i][L] for i in range(PRIMARY_PROBES)]).float()
        good=G.norm(dim=1)>EPS; G=G[good]
        if G.shape[0]==0:
            Q=torch.empty((HIDDEN,0),device=DEVICE,dtype=torch.float32); s=torch.empty(0,device=DEVICE)
        else:
            G=norm(G); _,s,Vh=torch.linalg.svd(G,full_matrices=False)
            tol=max(float(s[0])*SVD_REL_TOL,EPS); rank=int((s>tol).sum().item())
            Q=Vh[:rank].T.contiguous() if rank else torch.empty((HIDDEN,0),device=DEVICE,dtype=torch.float32)
        bases.append(Q); ranks.append(Q.shape[1]); singular.append(s.detach().cpu().tolist())
    return bases,ranks,singular

Q8,RANKS,SINGULAR=build_basis()
print("       P8 ranks:",RANKS)

def overlap(v):
    z=[]
    for L in range(N_LAYERS):
        q=Q8[L]; z.append(float((q.T@v[L]).square().sum()/v[L].square().sum().clamp_min(EPS)) if q.shape[1] else 0.)
    return z

def guard_vectors(raw):
    out=[]; tel=[]
    for L in range(N_LAYERS):
        u=raw[L].float(); q=Q8[L]; proj=q@(q.T@u) if q.shape[1] else torch.zeros_like(u)
        ug=u-proj; rawn=float(u.norm()); rem=float(ug.norm())
        if rem<EPS: raise RuntimeError(f"Guard annihilated actuator L{L}")
        ug=ug/rem; ov=float(proj.square().sum()/u.square().sum().clamp_min(EPS))
        cs=float(torch.dot(norm(u),ug).clamp(-1,1)); ang=math.degrees(math.acos(max(-1,min(1,cs))))
        post=float((q.T@ug).norm()) if q.shape[1] else 0.
        tel.append({"layer":L,"overlap_energy":ov,"retained_norm_fraction":rem/max(rawn,EPS),"raw_guard_cos":cs,"raw_guard_angle_deg":ang,"post_basis_overlap":post})
        out.append(ug)
    return torch.stack(out).contiguous(),tel

print(f"       overlap F={np.mean(overlap(FISHER)):.6f} B={np.mean(overlap(BANK)):.6f} R={np.mean(overlap(RESIDUAL)):.6f}")

# =================================================================================================
# SEASC EVALUATION — copied from working TEST167
# =================================================================================================
def blank_tel(): return [{"requested":[],"realized":[]} for _ in range(N_LAYERS)]
def hooks(vectors,sign,scale,tel,state=None):
    hs=[]
    for L in range(N_LAYERS):
        def mk(li):
            def hk(module,args,out):
                old=out[0] if isinstance(out,tuple) else out; rest=out[1:] if isinstance(out,tuple) else None; B=old.shape[0]
                direction=(float(sign)*vectors[li][None,:].expand(B,-1)).float().contiguous()
                req=IVME*ENVELOPE[li]*float(scale); dose=torch.full((B,),req,device=DEVICE,dtype=torch.float32)
                oldlast=old[:,-1,:].float(); new=seasc_ext.seasc_batch(old,direction,dose); newlast=new[:,-1,:].float()
                real=(newlast-oldlast).norm(dim=-1)/oldlast.norm(dim=-1).clamp_min(EPS)
                tel[li]["requested"].extend(dose.detach().cpu().tolist()); tel[li]["realized"].extend(real.detach().cpu().tolist())
                if state is not None: state[li]=newlast.detach().float().cpu()
                return new if rest is None else (new,)+rest
            return hk
        hs.append(layers[L].register_forward_hook(mk(L)))
    return hs

@torch.inference_mode()
def logp(context,continuation,v=None,sign=0,scale=0):
    ids,mask,plen=sequence(context,continuation); tel=blank_tel(); hs=hooks(v,sign,scale,tel) if v is not None else []
    try:o=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    logits=o.logits.float(); target=ids[:,plen:]; pred=logits[:,plen-1:-1,:]; lp=F.log_softmax(pred,-1)
    score=float(lp.gather(-1,target.unsqueeze(-1)).squeeze(-1).mean())
    del ids,mask,o,logits,target,pred,lp
    return score,tel

def arm(name,v,scale,items):
    cs,pm,mm,req,real=[],[],[],[],[]
    for x in items:
        pp,t1=logp(x["context"],x["pos"],v,+1,scale); pn,t2=logp(x["context"],x["neg"],v,+1,scale)
        np_,t3=logp(x["context"],x["pos"],v,-1,scale); nn,t4=logp(x["context"],x["neg"],v,-1,scale)
        a,b=pp-pn,np_-nn; cs.append(.5*(a-b)); pm.append(a); mm.append(b)
        for t in (t1,t2,t3,t4):
            for L in range(N_LAYERS): req+=t[L]["requested"]; real+=t[L]["realized"]
    return {"method":name,"scale":float(scale),"Csym":float(np.mean(cs)),"Csym_sd":float(np.std(cs,ddof=1)),"plus_margin":float(np.mean(pm)),"minus_margin":float(np.mean(mm)),"positive_items":int(np.sum(np.array(cs)>0)),"requested_mean":float(np.mean(req)),"realized_mean":float(np.mean(real)),"item_csym":[float(z) for z in cs]}

@torch.inference_mode()
def collateral_one(text,cont,v,scale,sign):
    ids,mask,plen=sequence(text,cont); o0=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    l0=o0.logits[:,plen-1:-1,:].float(); target=ids[:,plen:]; p0=F.log_softmax(l0,-1); n0=-p0.gather(-1,target.unsqueeze(-1)).squeeze(-1).mean()
    tel=blank_tel(); hs=hooks(v,sign,scale,tel)
    try:o1=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    l1=o1.logits[:,plen-1:-1,:].float(); p1=F.log_softmax(l1,-1); n1=-p1.gather(-1,target.unsqueeze(-1)).squeeze(-1).mean()
    kl=(p0.exp()*(p0-p1)).sum(-1).mean(); ans=(float(n1-n0),float(kl))
    del ids,mask,o0,o1,l0,l1,target,p0,p1,n0,n1,kl
    return ans

def collateral(name,v,scale,probes):
    plus,minus=[],[]
    for ctx,cont in probes:
        plus.append(collateral_one(ctx,cont,v,scale,+1)); minus.append(collateral_one(ctx,cont,v,scale,-1))
    kl=[x[1] for x in plus+minus]; dn=[abs(x[0]) for x in plus+minus]
    return {"method":name,"scale":float(scale),"kl_sym":float(np.mean(kl)),"abs_delta_nll_sym":float(np.mean(dn))}

def retention_assay(name,v,scale,probes):
    vals=[]
    for ctx,cont in probes:
        base,_=logp(ctx,cont,None,0,0)
        for sign in (+1,-1):
            s,_=logp(ctx,cont,v,sign,scale); vals.append(float(-s+base))
    return {"method":name,"scale":float(scale),"retention_delta_nll_sym":float(np.mean(vals)),"retention_abs_delta_nll_sym":float(np.mean(np.abs(vals)))}

@torch.inference_mode()
def states(context,v=None,scale=0,sign=0):
    e=tokenizer(chat(context),return_tensors="pt").to(DEVICE); cap=[None]*N_LAYERS; hs=[]
    if v is None:
        for L in range(N_LAYERS):
            def mk(li):
                def hk(m,a,o): cap[li]=(o[0] if isinstance(o,tuple) else o)[:,-1,:].detach().float().cpu()
                return hk
            hs.append(layers[L].register_forward_hook(mk(L)))
    else: hs=hooks(v,sign,scale,blank_tel(),cap)
    try:model(**e,use_cache=False,return_dict=True)
    finally:
        for h in hs:h.remove()
    del e
    return cap

def canal(v,scale,items):
    dp,dt,tot=[],[],[]
    for x in items:
        h0=states(x["context"])
        for s in (+1,-1):
            h1=states(x["context"],v,scale,s)
            for L in range(N_LAYERS):
                a=h0[L][0]; d=h1[L][0]-a; target=float(s)*v[L].detach().cpu().float()
                al=torch.dot(d,target); par=al*target; per=d-par; den=a.norm().clamp_min(EPS)
                tot.append(float(d.norm()/den)); dt.append(float(par.norm()/den)); dp.append(float(per.norm()/den))
    return {"D_perp":float(np.mean(dp)),"D_target":float(np.mean(dt)),"D_total":float(np.mean(tot)),"off_target_fraction":float(np.mean(dp)/max(np.mean(tot),EPS))}

# =================================================================================================
# FORGE — four independent development families
# =================================================================================================
FORGE={
"A":[
{"context":"A weather instrument and a field station are visible, with no vertical relation specified.","pos":" The weather instrument is above the field station.","neg":" The weather instrument is below the field station."},
{"context":"A suspended sensor and a factory platform are visible, with no vertical relation specified.","pos":" The suspended sensor is above the factory platform.","neg":" The suspended sensor is below the factory platform."},
{"context":"A balloon and a village road are visible, with no vertical relation specified.","pos":" The balloon is above the village road.","neg":" The balloon is below the village road."},
{"context":"A hanging lamp and a workbench are visible, with no vertical relation specified.","pos":" The hanging lamp is above the workbench.","neg":" The hanging lamp is below the workbench."}],
"B":[
{"context":"A marker and a cabinet are shown. Their relative height is unspecified.","pos":" The marker is higher than the cabinet.","neg":" The marker is lower than the cabinet."},
{"context":"A sign and a doorway are shown. Their relative height is unspecified.","pos":" The sign is higher than the doorway.","neg":" The sign is lower than the doorway."},
{"context":"A camera and a desk are shown. Their relative height is unspecified.","pos":" The camera is higher than the desk.","neg":" The camera is lower than the desk."},
{"context":"A light and a shelf are shown. Their relative height is unspecified.","pos":" The light is higher than the shelf.","neg":" The light is lower than the shelf."}],
"C":[
{"context":"Consider the vertical axis containing a drone and a wall.","pos":" The drone occupies the upper position relative to the wall.","neg":" The drone occupies the lower position relative to the wall."},
{"context":"Consider the vertical axis containing a flag and a gate.","pos":" The flag occupies the upper position relative to the gate.","neg":" The flag occupies the lower position relative to the gate."},
{"context":"Consider the vertical axis containing a bird and a bridge.","pos":" The bird occupies the upper position relative to the bridge.","neg":" The bird occupies the lower position relative to the bridge."},
{"context":"Consider the vertical axis containing a cloud and a ridge.","pos":" The cloud occupies the upper position relative to the ridge.","neg":" The cloud occupies the lower position relative to the ridge."}],
"D":[
{"context":"Two objects are present: a kite and a tower. No spatial ordering is given.","pos":" Vertically, the kite is over the tower.","neg":" Vertically, the kite is under the tower."},
{"context":"Two objects are present: a clock and a cabinet. No spatial ordering is given.","pos":" Vertically, the clock is over the cabinet.","neg":" Vertically, the clock is under the cabinet."},
{"context":"Two objects are present: a painting and a sofa. No spatial ordering is given.","pos":" Vertically, the painting is over the sofa.","neg":" Vertically, the painting is under the sofa."},
{"context":"Two objects are present: a symbol and a line. No spatial ordering is given.","pos":" Vertically, the symbol is over the line.","neg":" Vertically, the symbol is under the line."}]}

FORGE_NEUTRAL=[
("A ceramic vase rests beside a folded cloth."," The objects remain stationary."),
("A closed box sits beside a plain notebook."," The objects remain stationary."),
("A metal key lies beside a paper card."," The objects remain stationary."),
("A wooden tray rests near an empty glass."," The objects remain stationary.")]

def guarded_beta(beta):
    raw=beta_vector(beta); gv,t=guard_vectors(raw)
    return raw,gv,t

# =================================================================================================
# COARSE GUARDED SYNTHETIC SEARCH
# =================================================================================================
print("\n[5/18] Guard frozen. Starting GUARDED synthetic beta search...")
COARSE=[]; CACHE={}
for beta in COARSE_BETAS:
    raw,v,t=guarded_beta(beta); CACHE[round(float(beta),6)]=(raw,v,t)
    fam=[]
    for fn,items in FORGE.items():
        r=arm(f"BETA_{beta:+.2f}",v,FORGE_SCALE,items); fam.append(r["Csym"])
    c=collateral("x",v,FORGE_SCALE,FORGE_NEUTRAL)
    q=retention_assay("x",v,FORGE_SCALE,RETENTION)
    h=canal(v,FORGE_SCALE,FORGE["A"][:2])
    row={"beta":float(beta),"C_mean":float(np.mean(fam)),"C_worst":float(np.min(fam)),"C_sd":float(np.std(fam,ddof=1)),
         "A":fam[0],"B":fam[1],"C":fam[2],"D":fam[3],"KL":c["kl_sym"],"ret":q["retention_abs_delta_nll_sym"],
         "Dperp":h["D_perp"],"guard_overlap":float(np.mean([x["overlap_energy"] for x in t])),
         "guard_norm":float(np.mean([x["retained_norm_fraction"] for x in t]))}
    COARSE.append(row)
    print(f"       β={beta:+.2f} C={row['C_mean']:+.6f} worst={row['C_worst']:+.6f} sd={row['C_sd']:.6f} KL={row['KL']:.6f} Ret={row['ret']:.6f} Dp={row['Dperp']:.6f}")

# =================================================================================================
# COARSE VIABILITY / PARETO
# =================================================================================================
print("\n[6/18] Coarse viability / Pareto...")
CMAX=max(x["C_mean"] for x in COARSE)
BASE0=next(x for x in COARSE if abs(x["beta"])<1e-9)
VIABLE=[x for x in COARSE if x["C_mean"]>=CAUSAL_FLOOR*CMAX and x["C_worst"]>0]
SAFE=[x for x in VIABLE if x["KL"]<=KL_TOL*BASE0["KL"] and x["ret"]<=RET_TOL*BASE0["ret"] and x["Dperp"]<=DPERP_TOL*BASE0["Dperp"]]
POOL=SAFE if SAFE else VIABLE if VIABLE else COARSE

def dominates(a,b):
    no_worse=(a["C_mean"]>=b["C_mean"] and a["KL"]<=b["KL"] and a["ret"]<=b["ret"] and a["Dperp"]<=b["Dperp"] and a["C_sd"]<=b["C_sd"])
    strict=(a["C_mean"]>b["C_mean"] or a["KL"]<b["KL"] or a["ret"]<b["ret"] or a["Dperp"]<b["Dperp"] or a["C_sd"]<b["C_sd"])
    return no_worse and strict

PARETO=[x for x in POOL if not any(dominates(y,x) for y in POOL if y is not x)]
PARETO=sorted(PARETO,key=lambda x:(-x["C_mean"],x["KL"],x["ret"],x["Dperp"],x["C_sd"],abs(x["beta"])))
print("       Cmax:",f"{CMAX:+.6f}","| viable:",len(VIABLE),"| safe:",len(SAFE),"| Pareto:",[x["beta"] for x in PARETO])
CENTER=PARETO[0]["beta"] if PARETO else max(COARSE,key=lambda x:x["C_mean"])["beta"]
print("       coarse center:",f"{CENTER:+.3f}")

# =================================================================================================
# FINE SEARCH
# =================================================================================================
print("\n[7/18] Fine guarded search...")
fine=np.arange(CENTER-FINE_RADIUS,CENTER+FINE_RADIUS+FINE_STEP/2,FINE_STEP)
FINE=[]
for beta in sorted(set(round(float(x),6) for x in fine)):
    raw,v,t=guarded_beta(beta); fam=[arm("x",v,FORGE_SCALE,it)["Csym"] for it in FORGE.values()]
    c=collateral("x",v,FORGE_SCALE,FORGE_NEUTRAL); q=retention_assay("x",v,FORGE_SCALE,RETENTION); h=canal(v,FORGE_SCALE,FORGE["A"][:2])
    row={"beta":beta,"C_mean":float(np.mean(fam)),"C_worst":float(np.min(fam)),"C_sd":float(np.std(fam,ddof=1)),
         "KL":c["kl_sym"],"ret":q["retention_abs_delta_nll_sym"],"Dperp":h["D_perp"]}
    FINE.append(row)
    print(f"       β={beta:+.3f} C={row['C_mean']:+.6f} worst={row['C_worst']:+.6f} KL={row['KL']:.6f} Ret={row['ret']:.6f} Dp={row['Dperp']:.6f}")

FMAX=max(x["C_mean"] for x in FINE)
FVIABLE=[x for x in FINE if x["C_mean"]>=CAUSAL_FLOOR*FMAX and x["C_worst"]>0]
FSAFE=[x for x in FVIABLE if x["KL"]<=KL_TOL*BASE0["KL"] and x["ret"]<=RET_TOL*BASE0["ret"] and x["Dperp"]<=DPERP_TOL*BASE0["Dperp"]]
FPOOL=FSAFE if FSAFE else FVIABLE if FVIABLE else FINE

# =================================================================================================
# LOFO — leave-one-FORGE-family-out stability
# =================================================================================================
print("\n[8/18] LOFO stability...")
LOFO=[]
fine_betas=sorted(set(x["beta"] for x in FPOOL))
for held in FORGE.keys():
    rows=[]
    for beta in fine_betas:
        _,v,_=guarded_beta(beta)
        fam=[arm("x",v,FORGE_SCALE,items)["Csym"] for fn,items in FORGE.items() if fn!=held]
        rows.append((beta,float(np.mean(fam)),float(np.min(fam)),float(np.std(fam,ddof=1)) if len(fam)>1 else 0.))
    valid=[x for x in rows if x[2]>0]
    chosen=max(valid if valid else rows,key=lambda x:(x[1],-x[3],-abs(x[0])))
    LOFO.append({"held_out":held,"beta":chosen[0],"C_mean":chosen[1],"C_worst":chosen[2],"C_sd":chosen[3]})
    print(f"       hold {held} -> β={chosen[0]:+.3f} C={chosen[1]:+.6f} worst={chosen[2]:+.6f}")

# =================================================================================================
# MICRO STABILITY + FROZEN WINNER
# =================================================================================================
print("\n[9/18] Micro stability...")
def micro_score(beta):
    vals=[]
    for b in (beta-MICRO_DELTA,beta,beta+MICRO_DELTA):
        _,v,_=guarded_beta(b); fam=[arm("x",v,FORGE_SCALE,it)["Csym"] for it in FORGE.values()]
        vals.append(float(np.mean(fam)))
    curvature=vals[0]-2*vals[1]+vals[2]
    return vals,float(np.std(vals,ddof=1)),float(curvature),float(min(vals))

MICRO=[]
for x in FPOOL:
    vals,sd,curv,mn=micro_score(x["beta"])
    row={**x,"micro_minus":vals[0],"micro_center":vals[1],"micro_plus":vals[2],"micro_sd":sd,"curvature":curv,"micro_min":mn}
    MICRO.append(row)
    print(f"       β={x['beta']:+.3f} micro={vals[0]:+.5f}/{vals[1]:+.5f}/{vals[2]:+.5f} sd={sd:.6f} curv={curv:+.6f}")

# Predeclared lexicographic freeze: causal viability already enforced; then stability/collateral, tie -> |beta|.
WIN=min(MICRO,key=lambda x:(x["micro_sd"],x["Dperp"],x["KL"],x["ret"],-x["C_mean"],abs(x["beta"])))
FROZEN_BETA=float(WIN["beta"])
RAW_SYNTH,GUARD_SYNTH,GUARD_TEL=guarded_beta(FROZEN_BETA)

print("\n[10/18] FREEZE...")
print(f"       FROZEN β = {FROZEN_BETA:+.6f}")
print(f"       microSD={WIN['micro_sd']:.6f} | C={WIN['C_mean']:+.6f} | KL={WIN['KL']:.6f} | Ret={WIN['ret']:.6f} | Dperp={WIN['Dperp']:.6f}")
print("       LOFO betas:",[x["beta"] for x in LOFO])
print("       FINAL remains unopened.")

# =================================================================================================
# FROZEN ARMS
# =================================================================================================
FISHER_GUARD,_=guard_vectors(FISHER)
BANK_GUARD,_=guard_vectors(BANK)
PRIMARY={
"FISHER_RAW":FISHER,
"FISHER_GUARD":FISHER_GUARD,
"BANK_RAW":BANK,
"BANK_GUARD":BANK_GUARD,
"SYNTH_RAW":RAW_SYNTH,
"SYNTH_GUARD":GUARD_SYNTH}

# =================================================================================================
# HELD-OUT FINAL — first access after freeze
# =================================================================================================
FINAL=[
{"context":"A rescue drone and a coastal road are visible, with no vertical relation specified.","pos":" The rescue drone is above the coastal road.","neg":" The rescue drone is below the coastal road."},
{"context":"A hanging display and a service counter are visible, with no vertical relation specified.","pos":" The hanging display is above the service counter.","neg":" The hanging display is below the service counter."},
{"context":"A hawk and a cliff path are visible, with no vertical relation specified.","pos":" The hawk is above the cliff path.","neg":" The hawk is below the cliff path."},
{"context":"A suspended inspection camera and a machine table are visible, with no vertical relation specified.","pos":" The suspended inspection camera is above the machine table.","neg":" The suspended inspection camera is below the machine table."},
{"context":"A navigation beacon and a pier deck are visible, with no vertical relation specified.","pos":" The navigation beacon is above the pier deck.","neg":" The navigation beacon is below the pier deck."},
{"context":"A balloon and an orchard are visible, with no vertical relation specified.","pos":" The balloon is above the orchard.","neg":" The balloon is below the orchard."},
{"context":"A ceiling projector and a classroom table are visible, with no vertical relation specified.","pos":" The ceiling projector is above the classroom table.","neg":" The ceiling projector is below the classroom table."},
{"context":"A suspended detector and a laboratory bench are visible, with no vertical relation specified.","pos":" The suspended detector is above the laboratory bench.","neg":" The suspended detector is below the laboratory bench."},
{"context":"A radar antenna and an airport building are visible, with no vertical relation specified.","pos":" The radar antenna is above the airport building.","neg":" The radar antenna is below the airport building."},
{"context":"A hanging banner and a reception desk are visible, with no vertical relation specified.","pos":" The hanging banner is above the reception desk.","neg":" The hanging banner is below the reception desk."},
{"context":"A glider and a highway are visible, with no vertical relation specified.","pos":" The glider is above the highway.","neg":" The glider is below the highway."},
{"context":"A suspended microphone and a theatre floor are visible, with no vertical relation specified.","pos":" The suspended microphone is above the theatre floor.","neg":" The suspended microphone is below the theatre floor."},
{"context":"A warning lamp and a loading bay are visible, with no vertical relation specified.","pos":" The warning lamp is above the loading bay.","neg":" The warning lamp is below the loading bay."},
{"context":"A cloud and a canyon floor are visible, with no vertical relation specified.","pos":" The cloud is above the canyon floor.","neg":" The cloud is below the canyon floor."},
{"context":"A hanging sculpture and a museum bench are visible, with no vertical relation specified.","pos":" The hanging sculpture is above the museum bench.","neg":" The hanging sculpture is below the museum bench."},
{"context":"An observation balloon and a riverbank are visible, with no vertical relation specified.","pos":" The observation balloon is above the riverbank.","neg":" The observation balloon is below the riverbank."}]

FINAL_NEUTRAL=[
("A bottle stands beside a folded towel."," The scene contains ordinary stationary objects."),
("A tray rests beside a pair of gloves."," The scene contains ordinary stationary objects."),
("A plate sits beside a clean napkin."," The scene contains ordinary stationary objects."),
("A suitcase stands near a wooden chair."," The scene contains ordinary stationary objects."),
("A small radio rests beside a book."," The scene contains ordinary stationary objects."),
("A cardboard folder lies beside a metal pen."," The scene contains ordinary stationary objects.")]

print("\n[11/18] FINAL — FIRST ACCESS AFTER FREEZE...")
FINAL_CAUSAL=[]; FINAL_COLL=[]; FINAL_RET=[]
for scale in SCALES:
    print(f"\n       DOSE={scale:.3f}")
    for name,v in PRIMARY.items():
        r=arm(name,v,scale,FINAL); c=collateral(name,v,scale,FINAL_NEUTRAL); q=retention_assay(name,v,scale,RETENTION)
        FINAL_CAUSAL.append(r); FINAL_COLL.append(c); FINAL_RET.append(q)
        print(f"       {name:<14} C={r['Csym']:+.6f} P+={r['positive_items']:02d}/16 KL={c['kl_sym']:.6f} Ret|dNLL|={q['retention_abs_delta_nll_sym']:.6f}")

print("\n[12/18] FINAL hidden-state canalization...")
FINAL_HIDDEN=[]
for scale in SCALES:
    for name,v in PRIMARY.items():
        h=canal(v,scale,FINAL[:4]); FINAL_HIDDEN.append({"method":name,"scale":float(scale),**h})
        print(f"       {name:<14} s={scale:.3f} Dperp={h['D_perp']:.6f} Dtarget={h['D_target']:.6f} off={h['off_target_fraction']:.4f}")

# =================================================================================================
# BOOTSTRAP
# =================================================================================================
print("\n[13/18] Paired bootstrap...")
def get(rows,name,scale): return next(x for x in rows if x["method"]==name and abs(x["scale"]-scale)<1e-9)
def boot(a,b,n=20000,seed=168):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64); rng=np.random.default_rng(seed); means=np.empty(n)
    for i in range(n):
        ix=rng.integers(0,len(d),len(d)); means[i]=d[ix].mean()
    return {"delta":float(d.mean()),"ci_low":float(np.quantile(means,.025)),"ci_high":float(np.quantile(means,.975))}

CONTRASTS=[]
for scale in SCALES:
    sg=get(FINAL_CAUSAL,"SYNTH_GUARD",scale)
    for ref in ("SYNTH_RAW","FISHER_RAW","FISHER_GUARD","BANK_RAW","BANK_GUARD"):
        rr=get(FINAL_CAUSAL,ref,scale); z=boot(sg["item_csym"],rr["item_csym"],seed=SEED+int(scale*1000)+len(ref))
        CONTRASTS.append({"scale":scale,"contrast":"SYNTH_GUARD-"+ref,**z})
        print(f"       s={scale:.3f} SYNTH_GUARD-{ref:<12} ΔC={z['delta']:+.6f} [{z['ci_low']:+.6f},{z['ci_high']:+.6f}]")

# =================================================================================================
# PRIMARY SCALE TABLE
# =================================================================================================
print("\n[14/18] Primary mechanism table @0.50...")
TABLE=[]
for name in PRIMARY:
    c=get(FINAL_CAUSAL,name,PRIMARY_SCALE); k=get(FINAL_COLL,name,PRIMARY_SCALE); r=get(FINAL_RET,name,PRIMARY_SCALE); h=get(FINAL_HIDDEN,name,PRIMARY_SCALE)
    row={"method":name,"C":c["Csym"],"KL":k["kl_sym"],"Ret":r["retention_abs_delta_nll_sym"],"Dperp":h["D_perp"],"positive":c["positive_items"]}
    TABLE.append(row)
    print(f"       {name:<14} C={row['C']:+.6f} KL={row['KL']:.6f} Ret={row['Ret']:.6f} Dperp={row['Dperp']:.6f} P+={row['positive']}/16")

# =================================================================================================
# NULLSPACE CHECK
# =================================================================================================
print("\n[15/18] Synthetic nullspace verification...")
NULLCHECK=[]
for name,v in (("SYNTH_RAW",RAW_SYNTH),("SYNTH_GUARD",GUARD_SYNTH)):
    vals=[]
    for L in range(N_LAYERS):
        z=float((Q8[L].T@v[L]).norm()) if Q8[L].shape[1] else 0.; vals.append(z)
        NULLCHECK.append({"method":name,"layer":L,"Q_overlap":z})
    print(f"       {name:<12} mean Q overlap={np.mean(vals):.8e} max={np.max(vals):.8e}")

# =================================================================================================
# DECISION — no FINAL selection, only descriptive classification
# =================================================================================================
print("\n[16/18] TEST168 decision...")
SG=next(x for x in TABLE if x["method"]=="SYNTH_GUARD")
SR=next(x for x in TABLE if x["method"]=="SYNTH_RAW")
FG=next(x for x in TABLE if x["method"]=="FISHER_GUARD")
FR=next(x for x in TABLE if x["method"]=="FISHER_RAW")
guard_preserves=SG["C"]>=0.90*SR["C"] if SR["C"]>0 else SG["C"]>SR["C"]
guard_safer=(SG["KL"]<SR["KL"] and SG["Ret"]<SR["Ret"])
beats_fisher_causal=SG["C"]>FR["C"]
not_worse_fisher_kl=SG["KL"]<=FR["KL"]
if guard_preserves and guard_safer and beats_fisher_causal and not_worse_fisher_kl:
    DECISION="GUARDED_SYNTHETIC_VECTOR_ADVANTAGE_SUPPORTED"
elif guard_preserves and guard_safer:
    DECISION="GUARDED_SYNTHETIC_RETENTION_SIGNAL"
elif SG["C"]>FR["C"]:
    DECISION="SYNTHETIC_CAUSAL_SIGNAL_WITHOUT_RETENTION_ADVANTAGE"
else:
    DECISION="GUARDED_SYNTHETIC_ADVANTAGE_NOT_CONFIRMED"
print("       Guard preserves synthetic causal:",guard_preserves)
print("       Guard improves KL+Retention     :",guard_safer)
print("       Synthetic guard > Fisher causal :",beats_fisher_causal)
print("       Synthetic guard KL <= Fisher    :",not_worse_fisher_kl)
print("       DECISION:",DECISION)

# =================================================================================================
# SAVE
# =================================================================================================
print("\n[17/18] Saving...")
def scalar_rows(rows): return [{k:v for k,v in r.items() if not isinstance(v,(list,tuple,dict))} for r in rows]
def savecsv(name,rows):
    rows=scalar_rows(rows)
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

savecsv("COARSE_SEARCH.csv",COARSE); savecsv("FINE_SEARCH.csv",FINE); savecsv("LOFO.csv",LOFO); savecsv("MICRO.csv",MICRO)
savecsv("FINAL_CAUSAL.csv",FINAL_CAUSAL); savecsv("FINAL_COLLATERAL.csv",FINAL_COLL); savecsv("FINAL_RETENTION.csv",FINAL_RET)
savecsv("FINAL_HIDDEN.csv",FINAL_HIDDEN); savecsv("FINAL_CONTRASTS.csv",CONTRASTS); savecsv("PRIMARY_TABLE.csv",TABLE)
savecsv("SYNTH_GUARD_TELEMETRY.csv",GUARD_TEL); savecsv("NULLSPACE_CHECK.csv",NULLCHECK)

np.savez_compressed(ROOT/"TEST168_GUARDED_SYNTHETIC_VECTOR_FORGE.npz",
fisher=FISHER.detach().cpu().numpy(),bank=BANK.detach().cpu().numpy(),residual=RESIDUAL.detach().cpu().numpy(),
synthetic_raw=RAW_SYNTH.detach().cpu().numpy(),synthetic_guard=GUARD_SYNTH.detach().cpu().numpy(),
fisher_guard=FISHER_GUARD.detach().cpu().numpy(),bank_guard=BANK_GUARD.detach().cpu().numpy(),
frozen_beta=np.asarray([FROZEN_BETA],dtype=np.float32),envelope=np.asarray(ENVELOPE,dtype=np.float32))

REPORT={"test":"TEST 168","title":"GUARDED SYNTHETIC VECTOR FORGE","model":MODEL_ID,"gpu":torch.cuda.get_device_name(0),"seed":SEED,
"geometry":"Fisher + beta*(Bank residual norm / Fisher-Bank cosine)*Bank residual","guard":"P8 TEST167 activation nullspace",
"coarse_betas":COARSE_BETAS,"fine_step":FINE_STEP,"micro_delta":MICRO_DELTA,"forge_scale":FORGE_SCALE,
"frozen_beta":FROZEN_BETA,"decision":DECISION,"mean_cos_fisher_bank":float(np.mean(COS)),"mean_bank_residual_norm":float(np.mean(RESN)),
"retention_baseline_losses":RET_LOSSES,"coarse":COARSE,"fine":FINE,"lofo":LOFO,"micro":MICRO,
"final_causal":FINAL_CAUSAL,"final_collateral":FINAL_COLL,"final_retention":FINAL_RET,"final_hidden":FINAL_HIDDEN,
"contrasts":CONTRASTS,"primary_table":TABLE,"nullspace_check":NULLCHECK,
"contract":["TEST167 CUDA SEASC implementation retained.","TEST167 activation-gradient definition retained unchanged.",
"P8 guard is fixed before synthetic beta search.","Beta is searched only on FORGE development families.","FINAL never selects beta, guard basis, rank or thresholds.",
"Raw and guarded versions of the exact same frozen synthetic beta are compared in FINAL.",
"Fisher and Bank raw/guarded controls are retained.","LOFO and micro perturbation are development stability diagnostics.",
"SEASC envelope is unchanged.","No runtime beta controller, cosine gate, brake, cap or saturation is used.",
"FINAL is first accessed only after FROZEN_BETA is fixed."]}

with open(ROOT/"REPORT.json","w",encoding="utf-8") as f: json.dump(REPORT,f,indent=2,ensure_ascii=False)

print("\n[18/18] COMPLETE")
print("="*154); print("TEST 168 — GUARDED SYNTHETIC VECTOR FORGE — SUMMARY"); print("="*154)
print(f"Mean cos(Fisher,Bank): {np.mean(COS):+.6f}")
print(f"Mean Bank residual norm: {np.mean(RESN):.6f}")
print(f"FROZEN beta: {FROZEN_BETA:+.6f}")
print("LOFO betas:",[x["beta"] for x in LOFO])
print(f"Synthetic raw P8 overlap : {np.mean(overlap(RAW_SYNTH)):.6f}")
print(f"Synthetic guard Q overlap: {np.mean([x['Q_overlap'] for x in NULLCHECK if x['method']=='SYNTH_GUARD']):.8e}")
print("DECISION:",DECISION)
print("Saved:",ROOT)
print("="*154)
