# TEST 167 — ACTIVATION-SPACE RETENTION GUARD: NULLSPACE CAUSAL CORRIDOR
# AkbasCore 3.2 / SEASC — single-cell compact implementation
# TEST166 Fisher/Bank/Residual geometry + TEST147 principle transferred correctly to activation space.
# SEARCH/RETENTION basis is frozen before FINAL. FINAL never selects vectors, probe count, rank or thresholds.

import os,sys,math,json,csv,random,shutil,subprocess,gc
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from transformers import AutoTokenizer,AutoModelForCausalLM
from torch.utils.cpp_extension import load_inline

print("="*154); print("TEST 167 — ACTIVATION-SPACE RETENTION GUARD: NULLSPACE CAUSAL CORRIDOR"); print("AKBASCORE 3.2 / SEASC"); print("="*154)
subprocess.check_call([sys.executable,"-m","pip","install","-q","transformers>=4.45.0","accelerate>=0.34.0","ninja","pybind11"])
os.environ["MAX_JOBS"]="2"; os.environ["TOKENIZERS_PARALLELISM"]="false"
if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required.")
DEVICE=torch.device("cuda"); SEED=167
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"; N_LAYERS=20
IVME,SONUM,ZIRVE,TABAN=0.10,0.30,0.70,0.20
FISHER_ALPHA,FISHER_EPS,EPS=0.10,1e-8,1e-10
PROBE_COUNTS=[1,2,4,8]; PRIMARY_PROBES=8
SVD_REL_TOL=1e-5
SCALES=[0.125,0.25,0.50,0.75,1.00]; PRIMARY_SCALE=0.50
SYSTEM="You are a concise reasoning assistant. Use only the information in the prompt."
ROOT=Path("/content/TEST167_ACTIVATION_NULLSPACE_GUARD"); ROOT.mkdir(parents=True,exist_ok=True)
print("GPU:",torch.cuda.get_device_name(0)); print("Model:",MODEL_ID); print("Layers: L0-L19 | probes:",PROBE_COUNTS)

# =================================================================================================
# CUDA SEASC — same frozen-norm direct injection used by TEST166
# =================================================================================================
BUILD="/tmp/akbascore_test167"; shutil.rmtree(BUILD,ignore_errors=True); os.makedirs(BUILD,exist_ok=True)
CPP=r"""#include <torch/extension.h>
torch::Tensor seasc_batch_cuda(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose);
torch::Tensor seasc_batch(torch::Tensor hidden,torch::Tensor actuator,torch::Tensor dose){
TORCH_CHECK(hidden.is_cuda()&&actuator.is_cuda()&&dose.is_cuda(),"CUDA tensors required");
TORCH_CHECK(hidden.dim()==3&&actuator.dim()==2&&dose.dim()==1,"bad dimensions");
TORCH_CHECK(hidden.size(0)==actuator.size(0)&&hidden.size(0)==dose.size(0)&&hidden.size(2)==actuator.size(1),"shape mismatch");
return seasc_batch_cuda(hidden,actuator,dose);}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("seasc_batch",&seasc_batch,"TEST167 SEASC");}"""
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
AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,at::ScalarType::BFloat16,out.scalar_type(),"test167_seasc",[&]{k<scalar_t><<<nv,T,sb,stream>>>(out.data_ptr<scalar_t>(),act.data_ptr<float>(),d.data_ptr<float>(),B,S,H);});
C10_CUDA_KERNEL_LAUNCH_CHECK();return out;}"""
print("\n[1/16] Compiling SEASC...")
seasc_ext=load_inline(name="akbascore_test167_cuda",cpp_sources=CPP,cuda_sources=CUDA,functions=None,extra_cflags=["-O3","-std=c++17"],extra_cuda_cflags=["-O3","--use_fast_math"],with_cuda=True,build_directory=BUILD,verbose=False)
print("       OK")

# =================================================================================================
# MODEL
# =================================================================================================
print("\n[2/16] Loading frozen model...")
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
# SEARCH GEOMETRY — same independent SEARCH bank principle as TEST166
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

print("\n[3/16] Building Fisher / Bank / Residual geometry...")
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

# β=-0.5 retained as a predeclared residual-composite diagnostic from TEST165; no TEST167 selection.
def beta_vector(beta):
    q=[]
    for L in range(N_LAYERS):
        ratio=RESN[L]/max(COS[L],1e-8); q.append(norm(FISHER[L]+float(beta)*ratio*RESIDUAL[L]).float())
    return torch.stack(q).contiguous()
BETA_NEG05=beta_vector(-0.5)

# =================================================================================================
# RETENTION PROBES — distinct from SEARCH and FINAL.
# Chosen-token NLL gives a non-zero first derivative at vanilla baseline.
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
    # Critical: frozen parameters stay frozen. Grad graph is opened from detached input embeddings.
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
        loss.backward()
        grads=[]
        for L in range(N_LAYERS):
            if saved[L] is None or saved[L].grad is None: raise RuntimeError(f"No activation gradient at L{L}")
            # Injection acts on every token in TEST166 SEASC, so aggregate the gradient over sequence.
            # This is the exact first-order derivative for a shared additive actuator direction.
            g=saved[L].grad[0].float().sum(dim=0).detach()
            if not torch.isfinite(g).all(): raise RuntimeError(f"Nonfinite retention gradient L{L}")
            grads.append(g)
        lv=float(loss.detach())
    finally:
        for h in handles: h.remove()
        model.zero_grad(set_to_none=True)
    del ids,mask,embeds,out,logits,target,pred,lp,loss,saved
    return grads,lv

print("\n[4/16] Computing retention-sensitive activation gradients...")
PROBE_GRADS=[]; RET_LOSSES=[]
for i,(ctx,cont) in enumerate(RETENTION,1):
    g,l=activation_gradient(ctx,cont); PROBE_GRADS.append(g); RET_LOSSES.append(l)
    print(f"       probe {i}/8 | NLL={l:.6f} | mean ||g||={np.mean([float(x.norm()) for x in g]):.6e}")
    torch.cuda.empty_cache()

# =================================================================================================
# NULLSPACE BASIS
# Rows = normalized probe gradients. SVD gives orthonormal right-singular vectors in activation space.
# Rank is numerical only; no FINAL information participates.
# =================================================================================================
def build_basis(n):
    bases=[]; ranks=[]; singular=[]
    for L in range(N_LAYERS):
        G=torch.stack([PROBE_GRADS[i][L] for i in range(n)]).float()
        good=G.norm(dim=1)>EPS; G=G[good]
        if G.shape[0]==0:
            Q=torch.empty((HIDDEN,0),device=DEVICE,dtype=torch.float32); s=torch.empty(0,device=DEVICE)
        else:
            G=norm(G); _,s,Vh=torch.linalg.svd(G,full_matrices=False)
            tol=max(float(s[0])*SVD_REL_TOL,EPS); rank=int((s>tol).sum().item())
            Q=Vh[:rank].T.contiguous() if rank else torch.empty((HIDDEN,0),device=DEVICE,dtype=torch.float32)
        bases.append(Q); ranks.append(Q.shape[1]); singular.append(s.detach().cpu().tolist())
    return bases,ranks,singular

BASES={}
for n in PROBE_COUNTS:
    Q,r,s=build_basis(n); BASES[n]={"Q":Q,"rank":r,"singular":s}
    print(f"       P{n}: ranks={r} | mean={np.mean(r):.2f}")

def guard_vectors(raw,Q):
    out=[]; tel=[]
    for L in range(N_LAYERS):
        u=raw[L].float(); q=Q[L]
        proj=q@(q.T@u) if q.shape[1] else torch.zeros_like(u)
        ug=u-proj; rawnorm=float(u.norm()); rem=float(ug.norm())
        if rem<EPS: raise RuntimeError(f"Guard annihilated actuator at L{L}")
        ug=ug/rem
        overlap=float(proj.square().sum()/u.square().sum().clamp_min(EPS))
        cos=float(torch.dot(norm(u),ug).clamp(-1,1)); angle=math.degrees(math.acos(max(-1,min(1,cos))))
        post=float((q.T@ug).norm()) if q.shape[1] else 0.
        tel.append({"layer":L,"rank":q.shape[1],"overlap_energy":overlap,"retained_norm_fraction":rem/max(rawnorm,EPS),"raw_guard_cos":cos,"raw_guard_angle_deg":angle,"post_basis_overlap":post})
        out.append(ug)
    return torch.stack(out).contiguous(),tel

print("\n[5/16] Building RAW/GUARDED actuator families...")
RAW={"FISHER_RAW":FISHER,"BANK_RAW":BANK,"RESIDUAL_RAW":RESIDUAL,"BETA_NEG05_RAW":BETA_NEG05}
GUARDED={}; GUARD_TEL={}
for n in PROBE_COUNTS:
    for name,v in RAW.items():
        gn=name.replace("_RAW",f"_GUARD_P{n}"); gv,t=guard_vectors(v,BASES[n]["Q"])
        GUARDED[gn]=gv; GUARD_TEL[gn]=t
        print(f"       {gn:<24} overlap={np.mean([x['overlap_energy'] for x in t]):.6f} | norm={np.mean([x['retained_norm_fraction'] for x in t]):.6f} | angle={np.mean([x['raw_guard_angle_deg'] for x in t]):.3f}°")

# Explicit TEST167 hypothesis telemetry
def overlap(v,Q):
    vals=[]
    for L in range(N_LAYERS):
        q=Q[L]; vals.append(float((q.T@v[L]).square().sum()/v[L].square().sum().clamp_min(EPS)) if q.shape[1] else 0.)
    return vals
Q8=BASES[PRIMARY_PROBES]["Q"]
OVERLAPS={"FISHER":overlap(FISHER,Q8),"BANK":overlap(BANK,Q8),"RESIDUAL":overlap(RESIDUAL,Q8),"BETA_NEG05":overlap(BETA_NEG05,Q8)}
print("       P8 mean overlap F/R/B:",np.mean(OVERLAPS["FISHER"]),np.mean(OVERLAPS["RESIDUAL"]),np.mean(OVERLAPS["BANK"]))

# =================================================================================================
# SEASC EVALUATION
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
    ids,mask,plen=sequence(context,continuation); tel=blank_tel(); hs=[]
    if v is not None: hs=hooks(v,sign,scale,tel)
    try: o=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
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
    kl=(p0.exp()*(p0-p1)).sum(-1).mean()
    ans=(float(n1-n0),float(kl))
    del ids,mask,o0,o1,l0,l1,target,p0,p1,n0,n1,kl
    return ans

def collateral(name,v,scale,probes):
    plus,minus=[],[]
    for ctx,cont in probes:
        plus.append(collateral_one(ctx,cont,v,scale,+1)); minus.append(collateral_one(ctx,cont,v,scale,-1))
    kl=[x[1] for x in plus+minus]; dn=[abs(x[0]) for x in plus+minus]
    return {"method":name,"scale":float(scale),"kl_sym":float(np.mean(kl)),"abs_delta_nll_sym":float(np.mean(dn)),"kl_plus":float(np.mean([x[1] for x in plus])),"kl_minus":float(np.mean([x[1] for x in minus]))}

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
# DEVELOPMENT ASSAY — probe ablation only. Does NOT choose P; P8 was declared above.
# =================================================================================================
DEV=[
{"context":"A suspended lamp and a workshop table are visible, with no vertical relation specified.","pos":" The suspended lamp is above the workshop table.","neg":" The suspended lamp is below the workshop table."},
{"context":"A weather balloon and a farm road are visible, with no vertical relation specified.","pos":" The weather balloon is above the farm road.","neg":" The weather balloon is below the farm road."},
{"context":"A camera and a doorway are visible, with no vertical relation specified.","pos":" The camera is above the doorway.","neg":" The camera is below the doorway."},
{"context":"A kite and a stone wall are visible, with no vertical relation specified.","pos":" The kite is above the stone wall.","neg":" The kite is below the stone wall."}]
DEV_NEUTRAL=[
("A ceramic vase rests beside a folded cloth."," The objects remain stationary."),
("A closed box sits beside a plain notebook."," The objects remain stationary.")]

print("\n[6/16] Probe-ablation development assay @0.50...")
ABLATION=[]
for n in PROBE_COUNTS:
    for base in ("FISHER","RESIDUAL","BETA_NEG05"):
        rn=f"{base}_RAW"; gn=f"{base}_GUARD_P{n}"
        rv=RAW[rn]; gv=GUARDED[gn]
        rr=arm(rn,rv,PRIMARY_SCALE,DEV); gr=arm(gn,gv,PRIMARY_SCALE,DEV)
        rc=collateral(rn,rv,PRIMARY_SCALE,DEV_NEUTRAL); gc=collateral(gn,gv,PRIMARY_SCALE,DEV_NEUTRAL)
        row={"probes":n,"base":base,"raw_C":rr["Csym"],"guard_C":gr["Csym"],"delta_C":gr["Csym"]-rr["Csym"],"raw_KL":rc["kl_sym"],"guard_KL":gc["kl_sym"],"delta_KL":gc["kl_sym"]-rc["kl_sym"]}
        ABLATION.append(row); print(f"       P{n} {base:<10} C {rr['Csym']:+.5f}->{gr['Csym']:+.5f} | KL {rc['kl_sym']:.5f}->{gc['kl_sym']:.5f}")

# =================================================================================================
# TRUE RETENTION ASSAY — evaluates the actual probe continuation NLL under steering.
# Not used to build Q after this point; Q already frozen.
# =================================================================================================
print("\n[7/16] Retention-task assay...")
def retention_assay(name,v,scale,probes):
    vals=[]
    for ctx,cont in probes:
        ids,mask,plen=sequence(ctx,cont)
        with torch.inference_mode():
            o0=model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
            z0=o0.logits[:,plen-1:-1,:].float(); tar=ids[:,plen:]; p0=F.log_softmax(z0,-1)
            base=float(-p0.gather(-1,tar.unsqueeze(-1)).squeeze(-1).mean())
        for sign in (+1,-1):
            s,_=logp(ctx,cont,v,sign,scale); vals.append(float(-s-base))
        del ids,mask,o0,z0,tar,p0
    return {"method":name,"scale":float(scale),"retention_delta_nll_sym":float(np.mean(vals)),"retention_abs_delta_nll_sym":float(np.mean(np.abs(vals))),"per_sign_delta":[float(x) for x in vals]}

# =================================================================================================
# FREEZE
# Primary guard is P8 by design, not selected by DEV or FINAL.
# =================================================================================================
print("\n[8/16] FREEZE...")
PRIMARY={
"FISHER_RAW":FISHER,
"FISHER_GUARD":GUARDED["FISHER_GUARD_P8"],
"BANK_RAW":BANK,
"BANK_GUARD":GUARDED["BANK_GUARD_P8"],
"RESIDUAL_RAW":RESIDUAL,
"RESIDUAL_GUARD":GUARDED["RESIDUAL_GUARD_P8"],
"BETA_NEG05_RAW":BETA_NEG05,
"BETA_NEG05_GUARD":GUARDED["BETA_NEG05_GUARD_P8"]}
print("       Guard probes frozen: 8")
print("       Arms frozen:",list(PRIMARY.keys()))
print("       FINAL remains unopened.")

# =================================================================================================
# FINAL — first access after complete freeze
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

print("\n[9/16] FINAL — FIRST ACCESS AFTER FREEZE...")
FINAL_CAUSAL=[]; FINAL_COLL=[]; FINAL_RET=[]
for scale in SCALES:
    print(f"\n       DOSE={scale:.3f}")
    for name,v in PRIMARY.items():
        r=arm(name,v,scale,FINAL); c=collateral(name,v,scale,FINAL_NEUTRAL); q=retention_assay(name,v,scale,RETENTION)
        FINAL_CAUSAL.append(r); FINAL_COLL.append(c); FINAL_RET.append(q)
        print(f"       {name:<19} C={r['Csym']:+.6f} P+={r['positive_items']:02d}/16 KL={c['kl_sym']:.6f} Ret|dNLL|={q['retention_abs_delta_nll_sym']:.6f}")

print("\n[10/16] FINAL hidden-state canalization...")
FINAL_HIDDEN=[]
for scale in SCALES:
    for name,v in PRIMARY.items():
        h=canal(v,scale,FINAL[:4]); FINAL_HIDDEN.append({"method":name,"scale":float(scale),**h})
        print(f"       {name:<19} s={scale:.3f} Dperp={h['D_perp']:.6f} Dtarget={h['D_target']:.6f} off={h['off_target_fraction']:.4f}")

# =================================================================================================
# PAIRED RAW ↔ GUARD BOOTSTRAP
# =================================================================================================
print("\n[11/16] Paired bootstrap RAW↔GUARD...")
def get(rows,name,scale): return next(x for x in rows if x["method"]==name and x["scale"]==scale)
def boot(a,b,n=20000,seed=167):
    d=np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64); rng=np.random.default_rng(seed); means=np.empty(n)
    for i in range(n):
        ix=rng.integers(0,len(d),len(d)); means[i]=d[ix].mean()
    return {"delta":float(d.mean()),"ci_low":float(np.quantile(means,.025)),"ci_high":float(np.quantile(means,.975))}
CONTRASTS=[]
for scale in SCALES:
    for base in ("FISHER","BANK","RESIDUAL","BETA_NEG05"):
        raw=get(FINAL_CAUSAL,base+"_RAW",scale); guard=get(FINAL_CAUSAL,base+"_GUARD",scale)
        z=boot(guard["item_csym"],raw["item_csym"],seed=SEED+int(scale*1000)+len(base))
        row={"scale":float(scale),"base":base,"guard_minus_raw":z["delta"],"ci_low":z["ci_low"],"ci_high":z["ci_high"]}
        CONTRASTS.append(row)
        print(f"       {base:<10} s={scale:.3f} ΔC={z['delta']:+.6f} [{z['ci_low']:+.6f},{z['ci_high']:+.6f}]")

# =================================================================================================
# MECHANISM TABLE
# =================================================================================================
print("\n[12/16] Mechanism table @0.50...")
MECHANISM=[]
for base in ("FISHER","BANK","RESIDUAL","BETA_NEG05"):
    raw,guard=base+"_RAW",base+"_GUARD"
    rc,gc=get(FINAL_CAUSAL,raw,PRIMARY_SCALE),get(FINAL_CAUSAL,guard,PRIMARY_SCALE)
    rk,gk=get(FINAL_COLL,raw,PRIMARY_SCALE),get(FINAL_COLL,guard,PRIMARY_SCALE)
    rr,gr=get(FINAL_RET,raw,PRIMARY_SCALE),get(FINAL_RET,guard,PRIMARY_SCALE)
    rh,gh=get(FINAL_HIDDEN,raw,PRIMARY_SCALE),get(FINAL_HIDDEN,guard,PRIMARY_SCALE)
    ov=float(np.mean(OVERLAPS[base if base!="BETA_NEG05" else "BETA_NEG05"]))
    row={"base":base,"overlap":ov,"raw_C":rc["Csym"],"guard_C":gc["Csym"],"delta_C":gc["Csym"]-rc["Csym"],"raw_KL":rk["kl_sym"],"guard_KL":gk["kl_sym"],"delta_KL":gk["kl_sym"]-rk["kl_sym"],"raw_ret_abs_dNLL":rr["retention_abs_delta_nll_sym"],"guard_ret_abs_dNLL":gr["retention_abs_delta_nll_sym"],"raw_Dperp":rh["D_perp"],"guard_Dperp":gh["D_perp"]}
    MECHANISM.append(row)
    print(f"       {base:<10} overlap={ov:.5f} | C {row['raw_C']:+.5f}->{row['guard_C']:+.5f} | KL {row['raw_KL']:.5f}->{row['guard_KL']:.5f} | Ret {row['raw_ret_abs_dNLL']:.5f}->{row['guard_ret_abs_dNLL']:.5f}")

# =================================================================================================
# FIRST-ORDER CHECK: Q span should be numerically nulled after guard
# =================================================================================================
print("\n[13/16] Nullspace verification...")
NULLCHECK=[]
for base in ("FISHER","BANK","RESIDUAL","BETA_NEG05"):
    raw=RAW[base+"_RAW"]; guard=PRIMARY[base+"_GUARD"]
    for L in range(N_LAYERS):
        q=Q8[L]
        ro=float((q.T@raw[L]).norm()) if q.shape[1] else 0.; go=float((q.T@guard[L]).norm()) if q.shape[1] else 0.
        NULLCHECK.append({"base":base,"layer":L,"raw_Q_overlap":ro,"guard_Q_overlap":go})
    print(f"       {base:<10} raw={np.mean([x['raw_Q_overlap'] for x in NULLCHECK if x['base']==base]):.6e} guard={np.mean([x['guard_Q_overlap'] for x in NULLCHECK if x['base']==base]):.6e}")

# =================================================================================================
# DECISION — descriptive mechanistic rule, predeclared; no winner search.
# =================================================================================================
print("\n[14/16] TEST167 decision...")
res=next(x for x in MECHANISM if x["base"]=="RESIDUAL")
bet=next(x for x in MECHANISM if x["base"]=="BETA_NEG05")
res_guard_improves_ret=res["guard_ret_abs_dNLL"]<res["raw_ret_abs_dNLL"]
res_guard_improves_kl=res["guard_KL"]<res["raw_KL"]
res_guard_preserves_causal=res["guard_C"]>=0.90*res["raw_C"] if res["raw_C"]>0 else res["guard_C"]>res["raw_C"]
beta_guard_improves_ret=bet["guard_ret_abs_dNLL"]<bet["raw_ret_abs_dNLL"]
beta_guard_improves_kl=bet["guard_KL"]<bet["raw_KL"]
beta_guard_preserves_causal=bet["guard_C"]>=0.90*bet["raw_C"] if bet["raw_C"]>0 else bet["guard_C"]>bet["raw_C"]
if (res_guard_improves_ret and res_guard_improves_kl and res_guard_preserves_causal) or (beta_guard_improves_ret and beta_guard_improves_kl and beta_guard_preserves_causal):
    DECISION="RETENTION_SENSITIVE_OVERLAP_MECHANISM_SUPPORTED"
elif res_guard_improves_ret or beta_guard_improves_ret:
    DECISION="PARTIAL_RETENTION_OVERLAP_SIGNAL"
else:
    DECISION="RETENTION_OVERLAP_HYPOTHESIS_NOT_SUPPORTED"
print("       RESIDUAL: Ret↓",res_guard_improves_ret,"KL↓",res_guard_improves_kl,"C preserved",res_guard_preserves_causal)
print("       BETA-.5 : Ret↓",beta_guard_improves_ret,"KL↓",beta_guard_improves_kl,"C preserved",beta_guard_preserves_causal)
print("       DECISION:",DECISION)

# =================================================================================================
# SAVE
# =================================================================================================
print("\n[15/16] Saving...")
def scalar_rows(rows): return [{k:v for k,v in r.items() if not isinstance(v,(list,tuple,dict))} for r in rows]
def savecsv(name,rows):
    rows=scalar_rows(rows)
    if not rows:return
    keys=sorted(set().union(*[set(r) for r in rows]))
    with open(ROOT/name,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
savecsv("PROBE_ABLATION.csv",ABLATION); savecsv("FINAL_CAUSAL.csv",FINAL_CAUSAL); savecsv("FINAL_COLLATERAL.csv",FINAL_COLL)
savecsv("FINAL_RETENTION.csv",FINAL_RET); savecsv("FINAL_HIDDEN.csv",FINAL_HIDDEN); savecsv("FINAL_CONTRASTS.csv",CONTRASTS)
savecsv("MECHANISM.csv",MECHANISM); savecsv("NULLSPACE_CHECK.csv",NULLCHECK)
for name,t in GUARD_TEL.items(): savecsv("GUARD_"+name+".csv",t)

np.savez_compressed(ROOT/"TEST167_ACTIVATION_NULLSPACE_GUARD.npz",
fisher=FISHER.detach().cpu().numpy(),bank=BANK.detach().cpu().numpy(),residual=RESIDUAL.detach().cpu().numpy(),
beta_neg05=BETA_NEG05.detach().cpu().numpy(),fisher_guard=PRIMARY["FISHER_GUARD"].detach().cpu().numpy(),
bank_guard=PRIMARY["BANK_GUARD"].detach().cpu().numpy(),residual_guard=PRIMARY["RESIDUAL_GUARD"].detach().cpu().numpy(),
beta_neg05_guard=PRIMARY["BETA_NEG05_GUARD"].detach().cpu().numpy(),envelope=np.asarray(ENVELOPE,dtype=np.float32))

REPORT={"test":"TEST 167","title":"ACTIVATION-SPACE RETENTION GUARD: NULLSPACE CAUSAL CORRIDOR","model":MODEL_ID,"gpu":torch.cuda.get_device_name(0),
"seed":SEED,"guard_space":"layer-output activation space","retention_loss":"chosen-token continuation NLL","probe_counts":PROBE_COUNTS,
"primary_probes":PRIMARY_PROBES,"svd_relative_tolerance":SVD_REL_TOL,"decision":DECISION,"mean_cos_fisher_bank":float(np.mean(COS)),
"mean_bank_residual_norm":float(np.mean(RESN)),"retention_baseline_losses":RET_LOSSES,"primary_overlaps":OVERLAPS,
"probe_ablation":ABLATION,"final_causal":FINAL_CAUSAL,"final_collateral":FINAL_COLL,"final_retention":FINAL_RET,
"final_hidden":FINAL_HIDDEN,"final_contrasts":CONTRASTS,"mechanism":MECHANISM,"nullspace_check":NULLCHECK,
"contract":["Parameter-space gradients are never projected directly from activation vectors.",
"Retention-sensitive gradients are computed with respect to transformer layer outputs.",
"Model parameters remain frozen while autograd is opened through detached input embeddings.",
"Chosen-token NLL is used because baseline KL has zero first derivative at exact identity.",
"Probe gradients define a layer-local activation-space subspace.","The symmetric nullspace projection is identical for + and - steering.",
"Guarded vectors are renormalized only after projection.","P1/P2/P4/P8 are nested retention-probe prefixes.",
"P8 is declared primary before development and FINAL evaluation; ablation does not select the primary guard.",
"SEARCH prompts, retention probes, development prompts and FINAL prompts are separated.",
"FINAL is first accessed after geometry, guard basis and arms are frozen.","SEASC dose envelope is unchanged from TEST166.",
"No dynamic gate, brake, saturation, cap or runtime controller is used.","Raw injection and downstream hidden-state delta are measured as distinct quantities."]}
with open(ROOT/"REPORT.json","w",encoding="utf-8") as f: json.dump(REPORT,f,indent=2,ensure_ascii=False)

print("\n[16/16] COMPLETE")
print("="*154); print("TEST 167 — ACTIVATION-SPACE RETENTION GUARD — SUMMARY"); print("="*154)
print(f"Mean cos(Fisher,Bank): {np.mean(COS):+.6f}")
print(f"Mean Bank residual norm: {np.mean(RESN):.6f}")
print(f"P8 overlap Fisher : {np.mean(OVERLAPS['FISHER']):.6f}")
print(f"P8 overlap Bank   : {np.mean(OVERLAPS['BANK']):.6f}")
print(f"P8 overlap Resid. : {np.mean(OVERLAPS['RESIDUAL']):.6f}")
print(f"P8 overlap Beta-.5: {np.mean(OVERLAPS['BETA_NEG05']):.6f}")
print("DECISION:",DECISION); print("Saved:",ROOT); print("="*154)
