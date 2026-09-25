 
==========================================================================================================================================================
TEST 167 — ACTIVATION-SPACE RETENTION GUARD: NULLSPACE CAUSAL CORRIDOR
AKBASCORE 3.2 / SEASC
==========================================================================================================================================================
GPU: NVIDIA A100-SXM4-40GB
Model: Qwen/Qwen2.5-7B-Instruct
Layers: L0-L19 | probes: [1, 2, 4, 8]

[1/16] Compiling SEASC...
       OK

[2/16] Loading frozen model...

Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
WARNING:huggingface_hub.utils._http:Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

config.json: 100%

 663/663 [00:00<00:00, 83.2kB/s]
tokenizer_config.json: 100%

 7.30k/7.30k [00:00<00:00, 838kB/s]
vocab.json: 100%

 2.78M/2.78M [00:00<00:00, 38.8MB/s]
merges.txt: 100%

 1.67M/1.67M [00:00<00:00, 62.3MB/s]
tokenizer.json: 100%

 7.03M/7.03M [00:00<00:00, 124MB/s]
model.safetensors.index.json: 100%

 27.8k/27.8k [00:00<00:00, 1.56MB/s]
Download complete: : 

 13.1GB,  334kB/s  
Reconstruction complete: 100%

 15.2GB / 15.2GB, 27.7MB/s  
Fetching 4 files: 100%

 4/4 [04:07<00:00, 57.92s/it]
Loading weights: 100%

 339/339 [00:03<00:00, 101.29it/s]
generation_config.json: 100%

 243/243 [00:00<00:00, 31.3kB/s]
       hidden: 3584 dtype: torch.bfloat16

[3/16] Building Fisher / Bank / Residual geometry...
       pair 01/16
       pair 02/16
       pair 03/16
       pair 04/16
       pair 05/16
       pair 06/16
       pair 07/16
       pair 08/16
       pair 09/16
       pair 10/16
       pair 11/16
       pair 12/16
       pair 13/16
       pair 14/16
       pair 15/16
       pair 16/16
       L00 cos(F,B)=+0.957046 residual=0.289934
       L01 cos(F,B)=+0.945882 residual=0.324511
       L02 cos(F,B)=+0.923687 residual=0.383148
       L03 cos(F,B)=+0.905784 residual=0.423739
       L04 cos(F,B)=+0.827902 residual=0.560873
       L05 cos(F,B)=+0.787565 residual=0.616231
       L06 cos(F,B)=+0.783951 residual=0.620823
       L07 cos(F,B)=+0.851357 residual=0.524587
       L08 cos(F,B)=+0.877778 residual=0.479067
       L09 cos(F,B)=+0.880560 residual=0.473935
       L10 cos(F,B)=+0.868166 residual=0.496274
       L11 cos(F,B)=+0.863361 residual=0.504587
       L12 cos(F,B)=+0.872287 residual=0.488994
       L13 cos(F,B)=+0.866555 residual=0.499081
       L14 cos(F,B)=+0.869072 residual=0.494686
       L15 cos(F,B)=+0.870898 residual=0.491464
       L16 cos(F,B)=+0.860497 residual=0.509455
       L17 cos(F,B)=+0.861645 residual=0.507511
       L18 cos(F,B)=+0.766311 residual=0.642469
       L19 cos(F,B)=+0.772911 residual=0.634515
       mean cos=+0.860661 | mean residual=0.498294

[4/16] Computing retention-sensitive activation gradients...
       probe 1/8 | NLL=10.108150 | mean ||g||=5.017349e+00
       probe 2/8 | NLL=10.067118 | mean ||g||=4.504144e+00
       probe 3/8 | NLL=12.503087 | mean ||g||=3.515239e+00
       probe 4/8 | NLL=11.644833 | mean ||g||=3.911535e+00
       probe 5/8 | NLL=12.823627 | mean ||g||=5.401210e+00
       probe 6/8 | NLL=9.968038 | mean ||g||=5.083509e+00
       probe 7/8 | NLL=9.568528 | mean ||g||=4.566213e+00
       probe 8/8 | NLL=10.291623 | mean ||g||=5.011105e+00
       P1: ranks=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] | mean=1.00
       P2: ranks=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2] | mean=2.00
       P4: ranks=[4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4] | mean=4.00
       P8: ranks=[8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8] | mean=8.00

[5/16] Building RAW/GUARDED actuator families...
       FISHER_GUARD_P1          overlap=0.000159 | norm=0.999921 | angle=0.558°
       BANK_GUARD_P1            overlap=0.000320 | norm=0.999840 | angle=0.850°
       RESIDUAL_GUARD_P1        overlap=0.000510 | norm=0.999745 | angle=1.056°
       BETA_NEG05_GUARD_P1      overlap=0.000152 | norm=0.999924 | angle=0.535°
       FISHER_GUARD_P2          overlap=0.000631 | norm=0.999685 | angle=1.308°
       BANK_GUARD_P2            overlap=0.000751 | norm=0.999624 | angle=1.464°
       RESIDUAL_GUARD_P2        overlap=0.001074 | norm=0.999463 | angle=1.648°
       BETA_NEG05_GUARD_P2      overlap=0.000685 | norm=0.999658 | angle=1.238°
       FISHER_GUARD_P4          overlap=0.001286 | norm=0.999357 | angle=1.914°
       BANK_GUARD_P4            overlap=0.001495 | norm=0.999252 | angle=2.163°
       RESIDUAL_GUARD_P4        overlap=0.002050 | norm=0.998974 | angle=2.362°
       BETA_NEG05_GUARD_P4      overlap=0.001381 | norm=0.999309 | angle=1.982°
       FISHER_GUARD_P8          overlap=0.002564 | norm=0.998717 | angle=2.761°
       BANK_GUARD_P8            overlap=0.002598 | norm=0.998700 | angle=2.850°
       RESIDUAL_GUARD_P8        overlap=0.003997 | norm=0.997999 | angle=3.459°
       BETA_NEG05_GUARD_P8      overlap=0.002969 | norm=0.998514 | angle=2.953°
       P8 mean overlap F/R/B: 0.002564004802843556 0.00399738698033616 0.0025983900879509745

[6/16] Probe-ablation development assay @0.50...
       P1 FISHER     C +0.22413->+0.22225 | KL 0.89199->0.78607
       P1 RESIDUAL   C -0.16964->-0.17863 | KL 0.49238->0.45852
       P1 BETA_NEG05 C +0.24837->+0.28066 | KL 0.91950->0.77402
       P2 FISHER     C +0.22413->+0.22149 | KL 0.89199->0.83163
       P2 RESIDUAL   C -0.16964->-0.17192 | KL 0.49238->0.43820
       P2 BETA_NEG05 C +0.24837->+0.26995 | KL 0.91950->0.91141
       P4 FISHER     C +0.22413->+0.21729 | KL 0.89199->0.81741
       P4 RESIDUAL   C -0.16964->-0.18994 | KL 0.49238->0.40614
       P4 BETA_NEG05 C +0.24837->+0.26091 | KL 0.91950->0.85690
       P8 FISHER     C +0.22413->+0.22242 | KL 0.89199->0.62308
       P8 RESIDUAL   C -0.16964->-0.19460 | KL 0.49238->0.42419
       P8 BETA_NEG05 C +0.24837->+0.24970 | KL 0.91950->0.62925

[7/16] Retention-task assay...

[8/16] FREEZE...
       Guard probes frozen: 8
       Arms frozen: ['FISHER_RAW', 'FISHER_GUARD', 'BANK_RAW', 'BANK_GUARD', 'RESIDUAL_RAW', 'RESIDUAL_GUARD', 'BETA_NEG05_RAW', 'BETA_NEG05_GUARD']
       FINAL remains unopened.

[9/16] FINAL — FIRST ACCESS AFTER FREEZE...

       DOSE=0.125
       FISHER_RAW          C=+0.021360 P+=11/16 KL=0.020546 Ret|dNLL|=0.265679
       FISHER_GUARD        C=+0.016869 P+=08/16 KL=0.012542 Ret|dNLL|=0.153239
       BANK_RAW            C=-0.019009 P+=07/16 KL=0.020509 Ret|dNLL|=0.190811
       BANK_GUARD          C=-0.020266 P+=04/16 KL=0.017412 Ret|dNLL|=0.213528
       RESIDUAL_RAW        C=-0.063521 P+=03/16 KL=0.018749 Ret|dNLL|=0.118875
       RESIDUAL_GUARD      C=-0.062890 P+=02/16 KL=0.014809 Ret|dNLL|=0.173314
       BETA_NEG05_RAW      C=+0.039012 P+=13/16 KL=0.023137 Ret|dNLL|=0.238496
       BETA_NEG05_GUARD    C=+0.030946 P+=11/16 KL=0.017216 Ret|dNLL|=0.112422

       DOSE=0.250
       FISHER_RAW          C=+0.038252 P+=09/16 KL=0.111985 Ret|dNLL|=0.511684
       FISHER_GUARD        C=+0.046307 P+=12/16 KL=0.074688 Ret|dNLL|=0.341277
       BANK_RAW            C=-0.009292 P+=08/16 KL=0.079767 Ret|dNLL|=0.393203
       BANK_GUARD          C=-0.013924 P+=09/16 KL=0.063767 Ret|dNLL|=0.419060
       RESIDUAL_RAW        C=-0.106910 P+=03/16 KL=0.066622 Ret|dNLL|=0.256956
       RESIDUAL_GUARD      C=-0.102901 P+=04/16 KL=0.055324 Ret|dNLL|=0.391753
       BETA_NEG05_RAW      C=+0.067787 P+=12/16 KL=0.121999 Ret|dNLL|=0.521741
       BETA_NEG05_GUARD    C=+0.057808 P+=12/16 KL=0.075971 Ret|dNLL|=0.218916

       DOSE=0.500
       FISHER_RAW          C=+0.081662 P+=12/16 KL=0.591788 Ret|dNLL|=0.885298
       FISHER_GUARD        C=+0.080459 P+=11/16 KL=0.383201 Ret|dNLL|=0.727141
       BANK_RAW            C=+0.052945 P+=11/16 KL=0.427192 Ret|dNLL|=0.813059
       BANK_GUARD          C=+0.033716 P+=10/16 KL=0.303896 Ret|dNLL|=0.886783
       RESIDUAL_RAW        C=-0.141900 P+=03/16 KL=0.331058 Ret|dNLL|=0.617433
       RESIDUAL_GUARD      C=-0.131220 P+=04/16 KL=0.278403 Ret|dNLL|=0.710794
       BETA_NEG05_RAW      C=+0.111113 P+=12/16 KL=0.589741 Ret|dNLL|=0.926766
       BETA_NEG05_GUARD    C=+0.108322 P+=12/16 KL=0.379849 Ret|dNLL|=0.545423

       DOSE=0.750
       FISHER_RAW          C=+0.076932 P+=09/16 KL=1.272864 Ret|dNLL|=1.099586
       FISHER_GUARD        C=+0.079940 P+=08/16 KL=0.950289 Ret|dNLL|=0.927694
       BANK_RAW            C=+0.125663 P+=12/16 KL=1.161182 Ret|dNLL|=0.881530
       BANK_GUARD          C=+0.091744 P+=11/16 KL=0.882724 Ret|dNLL|=1.044894
       RESIDUAL_RAW        C=-0.003611 P+=10/16 KL=1.280674 Ret|dNLL|=0.911403
       RESIDUAL_GUARD      C=-0.008662 P+=08/16 KL=1.024306 Ret|dNLL|=1.044500
       BETA_NEG05_RAW      C=+0.117253 P+=12/16 KL=1.252748 Ret|dNLL|=1.153860
       BETA_NEG05_GUARD    C=+0.133951 P+=12/16 KL=0.979038 Ret|dNLL|=0.830175

       DOSE=1.000
       FISHER_RAW          C=+0.006126 P+=08/16 KL=1.850390 Ret|dNLL|=1.124184
       FISHER_GUARD        C=-0.018599 P+=06/16 KL=1.498133 Ret|dNLL|=1.099214
       BANK_RAW            C=+0.078608 P+=11/16 KL=2.466681 Ret|dNLL|=0.895298
       BANK_GUARD          C=+0.036016 P+=10/16 KL=1.814293 Ret|dNLL|=1.151714
       RESIDUAL_RAW        C=+0.116252 P+=12/16 KL=2.778490 Ret|dNLL|=1.091466
       RESIDUAL_GUARD      C=+0.131671 P+=14/16 KL=2.302049 Ret|dNLL|=1.209720
       BETA_NEG05_RAW      C=+0.067101 P+=11/16 KL=1.798567 Ret|dNLL|=1.200856
       BETA_NEG05_GUARD    C=+0.070630 P+=09/16 KL=1.493815 Ret|dNLL|=0.915108

[10/16] FINAL hidden-state canalization...
       FISHER_RAW          s=0.125 Dperp=0.038080 Dtarget=0.022946 off=0.8317
       FISHER_GUARD        s=0.125 Dperp=0.038101 Dtarget=0.022946 off=0.8310
       BANK_RAW            s=0.125 Dperp=0.041602 Dtarget=0.025634 off=0.8288
       BANK_GUARD          s=0.125 Dperp=0.040958 Dtarget=0.025277 off=0.8288
       RESIDUAL_RAW        s=0.125 Dperp=0.037980 Dtarget=0.018114 off=0.8829
       RESIDUAL_GUARD      s=0.125 Dperp=0.037429 Dtarget=0.017651 off=0.8850
       BETA_NEG05_RAW      s=0.125 Dperp=0.036037 Dtarget=0.020096 off=0.8468
       BETA_NEG05_GUARD    s=0.125 Dperp=0.036236 Dtarget=0.020332 off=0.8452
       FISHER_RAW          s=0.250 Dperp=0.074270 Dtarget=0.045803 off=0.8251
       FISHER_GUARD        s=0.250 Dperp=0.073554 Dtarget=0.045767 off=0.8224
       BANK_RAW            s=0.250 Dperp=0.081008 Dtarget=0.051369 off=0.8221
       BANK_GUARD          s=0.250 Dperp=0.078817 Dtarget=0.051039 off=0.8173
       RESIDUAL_RAW        s=0.250 Dperp=0.073723 Dtarget=0.036055 off=0.8777
       RESIDUAL_GUARD      s=0.250 Dperp=0.071986 Dtarget=0.034966 off=0.8797
       BETA_NEG05_RAW      s=0.250 Dperp=0.069830 Dtarget=0.040462 off=0.8370
       BETA_NEG05_GUARD    s=0.250 Dperp=0.070191 Dtarget=0.040599 off=0.8372
       FISHER_RAW          s=0.500 Dperp=0.145321 Dtarget=0.090372 off=0.8227
       FISHER_GUARD        s=0.500 Dperp=0.143988 Dtarget=0.090344 off=0.8207
       BANK_RAW            s=0.500 Dperp=0.158867 Dtarget=0.099713 off=0.8236
       BANK_GUARD          s=0.500 Dperp=0.154795 Dtarget=0.099071 off=0.8193
       RESIDUAL_RAW        s=0.500 Dperp=0.148323 Dtarget=0.071284 off=0.8794
       RESIDUAL_GUARD      s=0.500 Dperp=0.144400 Dtarget=0.069538 off=0.8795
       BETA_NEG05_RAW      s=0.500 Dperp=0.137269 Dtarget=0.080285 off=0.8343
       BETA_NEG05_GUARD    s=0.500 Dperp=0.137627 Dtarget=0.080773 off=0.8336
       FISHER_RAW          s=0.750 Dperp=0.214635 Dtarget=0.133646 off=0.8222
       FISHER_GUARD        s=0.750 Dperp=0.212840 Dtarget=0.133765 off=0.8201
       BANK_RAW            s=0.750 Dperp=0.234224 Dtarget=0.145252 off=0.8250
       BANK_GUARD          s=0.750 Dperp=0.228850 Dtarget=0.144061 off=0.8219
       RESIDUAL_RAW        s=0.750 Dperp=0.221988 Dtarget=0.104037 off=0.8822
       RESIDUAL_GUARD      s=0.750 Dperp=0.216545 Dtarget=0.101042 off=0.8829
       BETA_NEG05_RAW      s=0.750 Dperp=0.204252 Dtarget=0.119736 off=0.8341
       BETA_NEG05_GUARD    s=0.750 Dperp=0.204425 Dtarget=0.120407 off=0.8330
       FISHER_RAW          s=1.000 Dperp=0.282834 Dtarget=0.174650 off=0.8238
       FISHER_GUARD        s=1.000 Dperp=0.280722 Dtarget=0.174622 off=0.8220
       BANK_RAW            s=1.000 Dperp=0.307309 Dtarget=0.186292 off=0.8289
       BANK_GUARD          s=1.000 Dperp=0.300773 Dtarget=0.185225 off=0.8257
       RESIDUAL_RAW        s=1.000 Dperp=0.296367 Dtarget=0.134149 off=0.8860
       RESIDUAL_GUARD      s=1.000 Dperp=0.290108 Dtarget=0.130622 off=0.8867
       BETA_NEG05_RAW      s=1.000 Dperp=0.270294 Dtarget=0.157619 off=0.8352
       BETA_NEG05_GUARD    s=1.000 Dperp=0.270107 Dtarget=0.158282 off=0.8341

[11/16] Paired bootstrap RAW↔GUARD...
       FISHER     s=0.125 ΔC=-0.004491 [-0.016338,+0.007929]
       BANK       s=0.125 ΔC=-0.001257 [-0.011249,+0.008175]
       RESIDUAL   s=0.125 ΔC=+0.000631 [-0.009720,+0.011881]
       BETA_NEG05 s=0.125 ΔC=-0.008066 [-0.014891,-0.001819]
       FISHER     s=0.250 ΔC=+0.008054 [-0.004013,+0.019495]
       BANK       s=0.250 ΔC=-0.004632 [-0.015092,+0.005240]
       RESIDUAL   s=0.250 ΔC=+0.004009 [-0.013047,+0.021982]
       BETA_NEG05 s=0.250 ΔC=-0.009979 [-0.020688,+0.001352]
       FISHER     s=0.500 ΔC=-0.001204 [-0.015943,+0.012729]
       BANK       s=0.500 ΔC=-0.019229 [-0.030849,-0.007604]
       RESIDUAL   s=0.500 ΔC=+0.010680 [-0.009979,+0.031712]
       BETA_NEG05 s=0.500 ΔC=-0.002792 [-0.021229,+0.016166]
       FISHER     s=0.750 ΔC=+0.003008 [-0.013993,+0.021156]
       BANK       s=0.750 ΔC=-0.033919 [-0.049767,-0.015348]
       RESIDUAL   s=0.750 ΔC=-0.005051 [-0.032757,+0.022974]
       BETA_NEG05 s=0.750 ΔC=+0.016699 [-0.007375,+0.041182]
       FISHER     s=1.000 ΔC=-0.024725 [-0.045976,-0.002723]
       BANK       s=1.000 ΔC=-0.042592 [-0.073654,-0.009419]
       RESIDUAL   s=1.000 ΔC=+0.015419 [-0.013006,+0.044050]
       BETA_NEG05 s=1.000 ΔC=+0.003529 [-0.022963,+0.030855]

[12/16] Mechanism table @0.50...
       FISHER     overlap=0.00256 | C +0.08166->+0.08046 | KL 0.59179->0.38320 | Ret 0.88530->0.72714
       BANK       overlap=0.00260 | C +0.05294->+0.03372 | KL 0.42719->0.30390 | Ret 0.81306->0.88678
       RESIDUAL   overlap=0.00400 | C -0.14190->-0.13122 | KL 0.33106->0.27840 | Ret 0.61743->0.71079
       BETA_NEG05 overlap=0.00297 | C +0.11111->+0.10832 | KL 0.58974->0.37985 | Ret 0.92677->0.54542

[13/16] Nullspace verification...
       FISHER     raw=4.816193e-02 guard=6.842672e-08
       BANK       raw=4.972457e-02 guard=6.370278e-08
       RESIDUAL   raw=6.033063e-02 guard=8.030509e-08
       BETA_NEG05 raw=5.151610e-02 guard=7.633801e-08

[14/16] TEST167 decision...
       RESIDUAL: Ret↓ False KL↓ True C preserved True
       BETA-.5 : Ret↓ True KL↓ True C preserved True
       DECISION: RETENTION_SENSITIVE_OVERLAP_MECHANISM_SUPPORTED

[15/16] Saving...

[16/16] COMPLETE
==========================================================================================================================================================
TEST 167 — ACTIVATION-SPACE RETENTION GUARD — SUMMARY
==========================================================================================================================================================
Mean cos(Fisher,Bank): +0.860661
Mean Bank residual norm: 0.498294
P8 overlap Fisher : 0.002564
P8 overlap Bank   : 0.002598
P8 overlap Resid. : 0.003997
P8 overlap Beta-.5: 0.002969
DECISION: RETENTION_SENSITIVE_OVERLAP_MECHANISM_SUPPORTED
Saved: /content/TEST167_ACTIVATION_NULLSPACE_GUARD


==========================================================================================================================================================
