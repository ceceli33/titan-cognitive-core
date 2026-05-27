# =============================================================================
# 🔱 AKBAŞ CORE 0.2 | CONSTITUTIONAL STEERING ENGINE
# 5D Constitution × Core 7 Categories × Dynamic Parameter Matrix
# =============================================================================
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings, os, time
warnings.filterwarnings('ignore')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

try:
    import gradio as gr
except ImportError:
    os.system('pip install -q gradio')
    import gradio as gr

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_ID = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

# =============================================================================
# 📜 5D ETHICAL CONSTITUTION — V0_FINAL COORDINATES
# =============================================================================
V0_FINAL = {
    "d1_harm":     0.9228,
    "d2_honesty":  0.9372,
    "d3_autonomy": 0.8788,
    "d4_fairness": 0.9196,
    "d5_humility": 0.9096,
}
V0_MASTER = sum(V0_FINAL.values()) / len(V0_FINAL)

# =============================================================================
# 🌐 CORE 7 — Her kategori kendi parametre matrisini taşıyor
# temperature  : yaratıcılık vs determinizm dengesi
# top_k        : kelime havuzu genişliği
# top_p        : olasılık eşiği
# rep_penalty  : tekrar cezası
# max_tokens   : cevap uzunluğu
# =============================================================================
CORE_7 = {
    "TECHNICAL": {
        "keywords": [
            "engineering", "repair", "mechanical", "circuit", "fix",
            "installation", "wiring", "maintenance", "troubleshoot",
            "hardware", "component", "technical", "build", "voltage",
            "engine", "motor", "electric", "assembly", "calibration",
            "torque", "blueprint", "structural", "load", "material"
        ],
        "v0_profil": [0.88, 0.95, 0.85, 0.90, 0.88],
        "params": {
            # Teknik sorular: determinizm yüksek, yaratıcılık düşük
            "temperature":      0.30,
            "top_k":            30,
            "top_p":            0.85,
            "repetition_penalty": 1.6,
            "max_new_tokens":   300,
        }
    },
    "AGRICULTURE": {
        "keywords": [
            "agriculture", "crop", "soil", "harvest", "irrigation",
            "livestock", "farming", "fertilizer", "seed", "yield",
            "plantation", "greenhouse", "pest", "drought", "cultivate",
            "cattle", "poultry", "organic", "rotational", "compost",
            "pollination", "grazing", "arable", "tillage", "erosion"
        ],
        "v0_profil": [0.90, 0.95, 0.88, 0.92, 0.90],
        "params": {
            # Tarım: pratik bilgi, orta determinizm
            "temperature":      0.40,
            "top_k":            40,
            "top_p":            0.88,
            "repetition_penalty": 1.5,
            "max_new_tokens":   300,
        }
    },
    "HEALTH_MEDICINE": {
        "keywords": [
            "disease", "treatment", "medicine", "symptom", "nutrition",
            "health", "doctor", "diagnosis", "infection", "therapy",
            "anatomy", "biology", "pain", "chronic", "clinical",
            "pharmaceutical", "dosage", "pathology", "immunity", "vaccine",
            "metabolic", "neurological", "cardiac", "respiratory", "surgical"
        ],
        "v0_profil": [0.98, 1.00, 0.85, 0.90, 0.92],
        "params": {
            # Tıp: en yüksek determinizm — hata kabul edilemez
            "temperature":      0.20,
            "top_k":            20,
            "top_p":            0.80,
            "repetition_penalty": 1.7,
            "max_new_tokens":   300,
        }
    },
    "LAW_ADMINISTRATIVE": {
        "keywords": [
            "law", "legal", "court", "regulation", "official",
            "petition", "military", "jurisdiction", "rights", "statute",
            "compliance", "contract", "legislation", "administrative", "tax",
            "liability", "defendant", "plaintiff", "verdict", "appeal",
            "ordinance", "treaty", "constitution", "enforcement", "warrant"
        ],
        "v0_profil": [0.92, 1.00, 0.88, 0.95, 0.90],
        "params": {
            # Hukuk: kesinlik kritik, yaratıcılık istenmiyor
            "temperature":      0.25,
            "top_k":            25,
            "top_p":            0.82,
            "repetition_penalty": 1.7,
            "max_new_tokens":   300,
        }
    },
    "SOCIAL_PHILOSOPHY": {
        "keywords": [
            "ethics", "philosophy", "social", "psychology", "consciousness",
            "society", "culture", "morality", "identity", "behavior",
            "cognitive", "anthropology", "emotion", "belief", "value",
            "existential", "epistemology", "metaphysics", "ontology", "rhetoric",
            "discourse", "ideology", "paradigm", "perception", "reasoning"
        ],
        "v0_profil": [0.90, 0.92, 0.98, 0.88, 0.95],
        "params": {
            # Felsefe/Sosyal: yaratıcılık korunuyor — bu kategoride sanatçı taraf yaşamalı
            "temperature":      0.72,
            "top_k":            60,
            "top_p":            0.93,
            "repetition_penalty": 1.4,
            "max_new_tokens":   350,
        }
    },
    "ECONOMY": {
        "keywords": [
            "investment", "market", "economy", "inflation", "stock",
            "finance", "silver", "gold", "commodity", "portfolio",
            "crypto", "interest", "trading", "asset", "fiscal",
            "liquidity", "volatility", "hedge", "dividend", "equity",
            "monetary", "deficit", "yield", "derivative", "arbitrage"
        ],
        "v0_profil": [0.90, 0.95, 0.92, 0.95, 0.88],
        "params": {
            # Ekonomi: analitik ama esnek — piyasa yorumu yaratıcılık ister
            "temperature":      0.45,
            "top_k":            45,
            "top_p":            0.90,
            "repetition_penalty": 1.5,
            "max_new_tokens":   300,
        }
    },
    "SYSTEM_SOFTWARE": {
        "keywords": [
            "code", "algorithm", "software", "data", "ai",
            "function", "class", "api", "database", "framework",
            "machine learning", "neural", "model", "deploy", "backend",
            "frontend", "script", "compiler", "runtime", "library",
            "python", "c++", "debug", "refactor", "architecture",
            "microservice", "pipeline", "inference", "embedding", "vector"
        ],
        "v0_profil": [0.88, 0.95, 0.85, 0.88, 0.90],
        "params": {
            # Yazılım: determinizm yüksek ama mimari sorular esneklik ister
            "temperature":      0.35,
            "top_k":            35,
            "top_p":            0.87,
            "repetition_penalty": 1.6,
            "max_new_tokens":   350,
        }
    },
    "GENERAL": {
        "keywords": [],
        "v0_profil": [V0_MASTER] * 5,
        "params": {
            # Genel: dengeli merkez
            "temperature":      0.55,
            "top_k":            50,
            "top_p":            0.90,
            "repetition_penalty": 1.5,
            "max_new_tokens":   300,
        }
    },
}

# =============================================================================
# 🧭 5D CONSTITUTION ANCHOR SET
# =============================================================================
CONSTITUTION_ANCHORS = {
    "d1": ["safe",       "harmless",    "protective",  "secure",      "careful"],
    "d2": ["honest",     "accurate",    "truthful",    "transparent", "precise"],
    "d3": ["autonomous", "respectful",  "unbiased",    "free",        "neutral"],
    "d4": ["fair",       "just",        "equitable",   "balanced",    "impartial"],
    "d5": ["humble",     "aware",       "limited",     "uncertain",   "machine"],
}

# =============================================================================
# 📦 MODEL LOADING
# =============================================================================
print(f"🔧 Device: {DEVICE}")
print("📦 Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
    device_map='auto'
)
model.eval()
print("✅ Model loaded")

# =============================================================================
# ⚙️ AKBAŞ GRAFTING ENGINE
# =============================================================================
print("\n⚙️  AkbaşCore 0.2 grafting starting...")

with torch.no_grad():

    # PHASE 1: 5D Constitution Vectors
    dimension_vectors = {}
    for dim, words in CONSTITUTION_ANCHORS.items():
        tokens = tokenizer(words, return_tensors='pt', padding=True).to(DEVICE)
        vectors = model.model.embed_tokens(tokens['input_ids'])
        dimension_vectors[dim] = F.normalize(
            vectors.mean(dim=1).mean(dim=0), dim=0
        ).to(model.dtype)

    # PHASE 2: V0_FINAL Weighted Master Compass
    v0_values = list(V0_FINAL.values())
    master_compass = sum(
        v0_values[i] * list(dimension_vectors.values())[i]
        for i in range(5)
    ) / sum(v0_values)
    master_compass = F.normalize(master_compass, dim=0)
    print(f"🧭 Master compass ready | shape: {master_compass.shape}")

    # PHASE 3: Embedding Grafting
    raw = model.model.embed_tokens.weight.data
    similarity = (raw * master_compass).sum(dim=-1, keepdim=True)
    delta = torch.clamp(
        V0_MASTER * 0.80 * 0.3 * similarity * master_compass.unsqueeze(0),
        -0.15, 0.15
    )
    model.model.embed_tokens.weight.data = raw + delta
    print(f"📐 Embedding grafting complete | vocab: {raw.shape[0]}")

    # PHASE 4: Layer Weight Grafting (3 Zones)
    grafted = 0
    for idx, layer in enumerate(model.model.layers):
        kuvvet = 0.80 if idx < 8 else (0.40 if idx < 16 else 0.0)
        if kuvvet == 0.0:
            continue
        for proj in [layer.self_attn.q_proj, layer.self_attn.v_proj]:
            w = proj.weight.data
            alignment = (w @ master_compass)
            delta = torch.clamp(
                V0_MASTER * kuvvet * 0.3 * alignment.unsqueeze(1) * master_compass.unsqueeze(0),
                -0.15, 0.15
            )
            proj.weight.data = w + delta
        grafted += 1

    print(f"🔩 Layer grafting complete | {grafted} layers")

print(f"✅ AkbaşCore 0.2 active | V0_MASTER: {V0_MASTER:.4f}\n")

# =============================================================================
# 🔍 CATEGORY DETECTION — O(1)
# =============================================================================
def detect_category(question: str) -> tuple[str, dict]:
    q = question.lower()
    for category, data in CORE_7.items():
        if category == "GENERAL":
            continue
        for keyword in data["keywords"]:
            if keyword in q:
                return category, data
    return "GENERAL", CORE_7["GENERAL"]

# =============================================================================
# 🛡️ CONSTITUTIONAL POST-PROCESSING
# Gemini'nin fikri — ama kaba filtreleme değil, akıllı denetim
# =============================================================================
UNCERTAINTY_MARKERS = [
    "i'm not sure", "i am not sure", "i cannot be certain",
    "i don't know", "i do not know", "it's hard to say",
    "i'm unable to", "i am unable to"
]

HALLUCINATION_MARKERS = [
    "as an ai", "as a language model", "i apologize",
    "i must clarify", "i should mention that i"
]

def constitutional_check(answer: str, category: str) -> tuple[str, str]:
    a = answer.lower()
    flag = None

    for marker in HALLUCINATION_MARKERS:
        if marker in a:
            flag = "HALLUCINATION_MARKER"
            break

    # Tıp ve Hukuk kategorisinde belirsizlik işaretleri kritik
    if not flag and category in ("HEALTH_MEDICINE", "LAW_ADMINISTRATIVE"):
        for marker in UNCERTAINTY_MARKERS:
            if marker in a:
                flag = "UNCERTAINTY_IN_CRITICAL_CATEGORY"
                break

    status = f"✅ CONSTITUTIONAL" if not flag else f"⚠️ FLAGGED: {flag}"
    return answer, status

# =============================================================================
# 🧠 INFERENCE ENGINE
# =============================================================================
def akbas_inference(user_question):
    if not user_question.strip():
        return "Query cannot be empty.", ""

    category, cat_data = detect_category(user_question)
    p = cat_data["params"]
    profil_scalar = sum(cat_data["v0_profil"]) / len(cat_data["v0_profil"])

    full_prompt = f"<|user|>\n{user_question}</s>\n<|assistant|>\n"
    inputs = tokenizer(full_prompt, return_tensors='pt').to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=p["max_new_tokens"],
            do_sample=True,
            temperature=p["temperature"],
            top_k=p["top_k"],
            top_p=p["top_p"],
            repetition_penalty=p["repetition_penalty"],
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    input_len = inputs['input_ids'].shape[1]
    new_tokens = output_ids.shape[1] - input_len
    tok_per_sec = new_tokens / elapsed if elapsed > 0 else 0

    answer = tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True
    ).strip()

    answer, const_status = constitutional_check(answer, category)

    stats = (
        f"📂 {category} | "
        f"🌡️ temp={p['temperature']} | "
        f"⚖️ scalar={profil_scalar:.3f} | "
        f"{const_status} | "
        f"⏱️ {elapsed:.1f}s | "
        f"{tok_per_sec:.1f} tok/s"
    )

    return answer, stats

# =============================================================================
# 🖥️ GRADIO PANEL
# =============================================================================
css = """
.gradio-container { max-width: 100% !important; width: 100% !important; }
textarea, input[type="text"] { font-size: 16px !important; }
"""

with gr.Blocks(css=css, title="AkbaşCore 0.2") as akbas_panel:

    gr.Markdown("""
    ## 🔱 AkbaşCore 0.2 | Constitutional Steering Engine
    **5D Constitution × Core 7 Dynamic Parameters × Zero-Latency Grafting**
    """)

    with gr.Row():
        with gr.Column(scale=1):
            txt_question = gr.Textbox(
                label="Your Question",
                placeholder="Type your question here...",
                lines=5
            )
            btn_submit = gr.Button("🚀 Submit", variant="primary")
            btn_clear  = gr.Button("🗑️ Clear",  variant="secondary")

            gr.Markdown("""
            ### 📋 Core 7 — Dynamic Temperature
            | Category | Temp |
            |---|---|
            | 🔧 TECHNICAL | 0.30 |
            | 🌱 AGRICULTURE | 0.40 |
            | 🏥 HEALTH | 0.20 |
            | ⚖️ LAW | 0.25 |
            | 💭 PHILOSOPHY | 0.72 |
            | 📈 ECONOMY | 0.45 |
            | 💻 SOFTWARE | 0.35 |
            """)

        with gr.Column(scale=2):
            txt_output = gr.Textbox(
                label="AkbaşCore Output",
                lines=18,
                show_copy_button=True
            )
            txt_stats = gr.Textbox(
                label="System Statistics",
                lines=2,
                interactive=False
            )

    btn_submit.click(
        fn=akbas_inference,
        inputs=[txt_question],
        outputs=[txt_output, txt_stats]
    )
    btn_clear.click(
        fn=lambda: ("", "", ""),
        outputs=[txt_question, txt_output, txt_stats]
    )

akbas_panel.launch(debug=False, share=True)
