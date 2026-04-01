"""
Udyam NIC Code Predictor — Inference v2
=========================================
Fixes:
  1. Division lookup uses NORMALIZED keys matching dataset
  2. Confidence threshold → asks ONE clarifying question if unsure
  3. Clarification is keyword-based, not another model call (zero token waste)
  4. Clean report display
"""

import tensorflow as tf
import numpy as np
import json
import sys

# =====================================================
# 1. RE-DEFINE TRANSFORMER (must match training)
# =====================================================

@tf.keras.utils.register_keras_serializable(package="UdyamNIC")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim    = embed_dim
        self.num_heads    = num_heads
        self.ff_dim       = ff_dim
        self.dropout_rate = dropout

        self.att   = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn   = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="gelu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, training=False):
        attn = self.att(x, x)
        attn = self.drop1(attn, training=training)
        x    = self.norm1(x + attn)
        ffn  = self.ffn(x)
        ffn  = self.drop2(ffn, training=training)
        return self.norm2(x + ffn)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim":    self.ff_dim,
            "dropout":   self.dropout_rate,
        })
        return config


# =====================================================
# 2. UDYAM GUIDANCE — keys match dataset division names
# =====================================================

UDYAM_GUIDANCE = {
    "Agriculture": {
        "portal":       "PM-KISAN / Kisan Suvidha Portal",
        "registration": "PM-KISAN (pmkisan.gov.in)",
        "schemes":      ["PM-KISAN", "PMFBY Crop Insurance", "Kisan Credit Card", "MIDH"],
        "compliance":   "APEDA registration if exporting",
    },
    "Manufacturing": {
        "portal":       "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["PMEGP", "Mudra Yojana", "CGTMSE", "PLI Scheme", "ZED Scheme"],
        "compliance":   "GST, Factory License, BIS Certification",
    },
    "Construction": {
        "portal":       "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["PMEGP", "Mudra Yojana", "PMAY"],
        "compliance":   "RERA Registration, Labour License, GST",
    },
    "Trade": {
        "portal":       "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["Mudra Yojana", "Stand Up India", "TReDS"],
        "compliance":   "GST Registration, Shop & Establishment License",
    },
    "Transport": {
        "portal":       "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["Mudra Yojana", "PM Gati Shakti"],
        "compliance":   "RTO Permit, National Transit Pass, GST",
    },
    "Hospitality": {
        "portal":       "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["Mudra Yojana", "PMEGP", "SWADESH DARSHAN"],
        "compliance":   "FSSAI Food License, GST, Shop & Establishment License",
    },
    "IT Services": {
        "portal":       "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["Startup India", "Digital India", "MEITY Schemes"],
        "compliance":   "GST Registration, Shop & Establishment License",
    },
    "Finance": {
        "portal":       "RBI / SEBI / IRDAI",
        "registration": "RBI Registration (for lending activities)",
        "schemes":      ["Stand Up India", "CGTMSE"],
        "compliance":   "RBI NBFC License, SEBI Registration, IRDAI",
    },
    "Real Estate": {
        "portal":       "RERA Portal (state-specific)",
        "registration": "RERA (state portal)",
        "schemes":      ["PMAY", "NHB Schemes"],
        "compliance":   "RERA Registration, GST",
    },
    "Professional": {
        "portal":       "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["Startup India", "Mudra Yojana"],
        "compliance":   "Professional Tax, GST",
    },
    "Education": {
        "portal":       "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["PMKVY", "NSDC Skill Development", "Samagra Shiksha"],
        "compliance":   "AICTE/UGC Approval, NSDC Accreditation",
    },
    "Healthcare": {
        "portal":       "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["Ayushman Bharat", "PLI Medical Devices", "PMJAY"],
        "compliance":   "NABH, CDSCO, Drug License, PCB",
    },
    "Services": {
        "portal":       "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["Mudra Yojana", "PMEGP", "PM SVANidhi"],
        "compliance":   "GST, Shop & Establishment License",
    },
    "Energy": {
        "portal":       "Udyam Registration Portal",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["Solar Subsidy (PM Surya Ghar)", "Green Hydrogen Mission", "REWA"],
        "compliance":   "CEA Guidelines, DISCOM Grid Approval",
    },
    "Mining": {
        "portal":       "IBM (ibm.gov.in)",
        "registration": "IBM Clearance",
        "schemes":      ["Mineral Exploration Trust", "NMDC Schemes"],
        "compliance":   "Mining Lease, IBM Clearance, PCB Consent",
    },
}


# =====================================================
# 3. CLARIFICATION — keyword-based, zero model cost
# =====================================================

# When confidence is low, ask ONE question to disambiguate
# Maps ambiguous keywords → clarifying question + refinement hints
AMBIGUOUS_PATTERNS = [
    {
        "keywords": ["repair", "fix", "service", "maintain"],
        "question": "Do you repair vehicles/bikes or electronic items or other equipment?",
        "options":  ["vehicles/bikes", "electronics/mobiles", "machinery/equipment"],
    },
    {
        "keywords": ["sell", "shop", "store", "dukan", "retail"],
        "question": "What do you mainly sell? (e.g. food, clothes, medicine, electronics, hardware)",
        "options":  [],
    },
    {
        "keywords": ["make", "manufacture", "produce", "banane"],
        "question": "What exactly do you make? Please name the product (e.g. soap, bags, furniture)",
        "options":  [],
    },
    {
        "keywords": ["farm", "grow", "cultivate", "kheti"],
        "question": "What crop or animal do you work with?",
        "options":  [],
    },
]

CONFIDENCE_THRESHOLD = 0.45  # below this → ask clarification


def needs_clarification(text: str, confidence: float) -> dict | None:
    """Returns clarification question dict if needed, else None."""
    if confidence >= CONFIDENCE_THRESHOLD:
        return None
    text_lower = text.lower()
    for pattern in AMBIGUOUS_PATTERNS:
        if any(kw in text_lower for kw in pattern["keywords"]):
            return pattern
    return None


# =====================================================
# 4. LOAD MODEL + VOCAB + LABELS
# =====================================================

print("\nLoading Udyam NIC Predictor Engine...")

try:
    model = tf.keras.models.load_model("models/udyam_nic_model.keras")

    with open("models/label_map.json") as f:
        label_map = json.load(f)

    nic_map    = label_map["nic"]
    div_map    = label_map["division"]
    nic_labels = label_map["nic_labels"]

    with open("models/vectorizer_vocab.json") as f:
        vocab_config = json.load(f)

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_config["max_vocab"],
        output_sequence_length=vocab_config["seq_len"],
        standardize="lower_and_strip_punctuation",
    )
    vectorizer.set_vocabulary(vocab_config["vocabulary"])
    print("Engine ready.\n")

except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have run: python build_dataset.py && python train_model.py first")
    sys.exit(1)


# =====================================================
# 5. PREDICTION FUNCTION
# =====================================================

def predict(text: str, top_k: int = 3) -> dict:
    vec = vectorizer(tf.constant([text]))
    preds = model(vec, training=False)
    preds = [p.numpy() for p in preds]

    nic_probs = preds[0][0]
    div_probs = preds[1][0]

    top_nic_idx = np.argsort(nic_probs)[::-1][:top_k]
    div_idx     = np.argmax(div_probs)

    top_nics = []
    for idx in top_nic_idx:
        code  = nic_map[str(idx)]
        label = nic_labels.get(code, "—")
        top_nics.append({
            "nic_code":   code,
            "nic_label":  label,
            "confidence": float(nic_probs[idx]),
        })

    division = div_map[str(div_idx)]
    guidance = UDYAM_GUIDANCE.get(division, {
        "portal":       "udyamregistration.gov.in",
        "registration": "udyamregistration.gov.in",
        "schemes":      ["Mudra Yojana", "PMEGP"],
        "compliance":   "GST, Shop & Establishment License",
    })

    return {
        "top_nics":       top_nics,
        "division":       division,
        "div_confidence": float(div_probs[div_idx]),
        "guidance":       guidance,
    }


# =====================================================
# 6. PRINT REPORT
# =====================================================

def print_report(text: str, clarification_suffix: str = ""):
    full_text = text + " " + clarification_suffix if clarification_suffix else text
    result    = predict(full_text.strip(), top_k=3)
    best      = result["top_nics"][0]
    div       = result["division"]
    guide     = result["guidance"]
    conf      = best["confidence"]

    print("\n" + "=" * 62)
    print(f"  Business : {text[:55]}")
    print("=" * 62)
    print(f"  NIC Code    : {best['nic_code']}")
    print(f"  Activity    : {best['nic_label']}")
    print(f"  Sector      : {div}")
    print(f"  Confidence  : {conf * 100:.1f}%")
    print("-" * 62)
    print(f"  Register At : {guide.get('registration', '—')}")
    print(f"  Portal      : {guide.get('portal', '—')}")
    print(f"  Compliance  : {guide.get('compliance', '—')}")
    schemes = guide.get('schemes', [])
    if schemes:
        print(f"  Schemes     : {', '.join(schemes)}")
    print("-" * 62)

    if len(result["top_nics"]) > 1:
        print("  Other possible codes:")
        for alt in result["top_nics"][1:]:
            print(f"    {alt['nic_code']} — {alt['nic_label'][:40]} ({alt['confidence']*100:.1f}%)")
    print("=" * 62)

    return result


# =====================================================
# 7. TEST CASES
# =====================================================

TEST_CASES = [
    "i make soap at home",
    "we grow rice",
    "i am a tailor",
    "we run a beauty salon",
    "truck transport business",
    "agarbatti making unit",
    "i make gold jewellery",
]

print("Running test cases...\n")
for t in TEST_CASES:
    print_report(t)


# =====================================================
# 8. INTERACTIVE MODE WITH CLARIFICATION
# =====================================================

print("\n" + "=" * 62)
print("  Udyam NIC Predictor — INTERACTIVE MODE")
print("  Type 'exit' to quit")
print("=" * 62)

try:
    while True:
        sys.stdout.write("\nDescribe your business: ")
        sys.stdout.flush()
        user_input = sys.stdin.readline().strip()

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            print("\nThank you. Visit udyamregistration.gov.in to register.")
            break

        # Quick predict to check confidence
        quick = predict(user_input, top_k=1)
        best_conf = quick["top_nics"][0]["confidence"]

        clarify = needs_clarification(user_input, best_conf)

        if clarify:
            print(f"\n  [Confidence {best_conf*100:.0f}%] Let me ask one quick question:")
            print(f"  >> {clarify['question']}")
            if clarify["options"]:
                for i, opt in enumerate(clarify["options"], 1):
                    print(f"     {i}. {opt}")
            sys.stdout.write("  Your answer: ")
            sys.stdout.flush()
            extra = sys.stdin.readline().strip()
            print_report(user_input, clarification_suffix=extra)
        else:
            print_report(user_input)

except KeyboardInterrupt:
    print("\n\nExiting...")
except EOFError:
    print("\n\nInput closed.")