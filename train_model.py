"""
Udyam NIC Code Predictor — Training Script v2
===============================================
Key improvements:
  - Loads expanded v2 dataset
  - Division labels normalized to match UDYAM_GUIDANCE keys
  - Larger model (embed_dim=192, 3 transformer layers)
  - Label smoothing for better generalization
  - Cosine LR scheduler 
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json, os

print("TensorFlow version:", tf.__version__)

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Metal/GPU Setup for Mac
if tf.config.list_physical_devices('GPU'):
    print("GPU (MPS/Metal) detected.")
else:
    print("Warning: No GPU detected, training on CPU.")


# =====================================================
# 1. LOAD DATA
# =====================================================

df = pd.read_csv("data/industries.csv")
df = df.drop_duplicates(subset=["text", "nic_code"])
df["nic_code"] = df["nic_code"].astype(str)

print(f"\nLoaded {len(df)} samples")
print(f"Unique NIC codes : {df['nic_code'].nunique()}")
print(f"Unique divisions : {df['division'].nunique()} → {sorted(df['division'].unique())}")

texts     = df["text"].astype(str).values.astype("U")   # Unicode string dtype
nic_codes = df["nic_code"].astype(str).values.astype("U")
divisions = df["division"].astype(str).values.astype("U")

# =====================================================
# 2. ENCODE LABELS
# =====================================================

nic_encoder = LabelEncoder()
div_encoder = LabelEncoder()

y_nic = nic_encoder.fit_transform(nic_codes)
y_div = div_encoder.fit_transform(divisions)

num_nic = len(nic_encoder.classes_)
num_div = len(div_encoder.classes_)

print(f"\nNIC classes  : {num_nic}")
print(f"Division classes: {num_div} → {list(div_encoder.classes_)}")

# Save label maps
label_map = {
    "nic": {str(i): c for i, c in enumerate(nic_encoder.classes_)},
    "division": {str(i): c for i, c in enumerate(div_encoder.classes_)},
}

nic_label_map = {}
for _, row in df[["nic_code", "nic_label"]].drop_duplicates().iterrows():
    nic_label_map[str(row["nic_code"])] = row["nic_label"]
label_map["nic_labels"] = nic_label_map

# Save division→guidance lookup separately
div_nic_map = {}
for _, row in df[["nic_code", "division"]].drop_duplicates().iterrows():
    div_nic_map[str(row["nic_code"])] = row["division"]
label_map["nic_to_division"] = div_nic_map

with open("models/label_map.json", "w") as f:
    json.dump(label_map, f, indent=2, ensure_ascii=False)

print("Label maps saved → models/label_map.json")

# =====================================================
# 3. TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, yn_train, yn_test, yd_train, yd_test = train_test_split(
    texts, y_nic, y_div, test_size=0.15, random_state=42, stratify=y_nic
)

# Convert to categorical for label smoothing support
yn_train = tf.keras.utils.to_categorical(yn_train, num_nic)
yn_test  = tf.keras.utils.to_categorical(yn_test, num_nic)
yd_train = tf.keras.utils.to_categorical(yd_train, num_div)
yd_test  = tf.keras.utils.to_categorical(yd_test, num_div)


print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

# =====================================================
# 4. TEXT VECTORIZATION
# =====================================================

MAX_VOCAB = 15000
SEQ_LEN   = 50
EMBED_DIM = 192

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_VOCAB,
    output_sequence_length=SEQ_LEN,
    standardize="lower_and_strip_punctuation",
    name="text_vectorizer",
)

X_train = np.array(X_train, dtype=str)
X_test  = np.array(X_test, dtype=str)

vectorizer.adapt(X_train)

X_train_vec = vectorizer(X_train)
X_test_vec  = vectorizer(X_test)

vocab_config = {
    "vocabulary": vectorizer.get_vocabulary(),
    "max_vocab": MAX_VOCAB,
    "seq_len": SEQ_LEN,
}
with open("models/vectorizer_vocab.json", "w") as f:
    json.dump(vocab_config, f)

print(f"Vocabulary size: {len(vocab_config['vocabulary'])}")

# =====================================================
# 5. TRANSFORMER BLOCK
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
# 6. BUILD MODEL — larger & deeper
# =====================================================

def build_model(vocab_size, embed_dim, seq_len, num_heads, ff_dim, num_nic, num_div):
    inputs = tf.keras.Input(shape=(seq_len,), name="token_ids")

    tok_emb = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim, name="token_embedding")(inputs)
    pos_emb = tf.keras.layers.Embedding(
        input_dim=seq_len, output_dim=embed_dim, name="position_embedding")(tf.range(seq_len))

    x = tok_emb + pos_emb
    x = tf.keras.layers.Dropout(0.1)(x)

    x = TransformerBlock(embed_dim, num_heads, ff_dim, name="transformer_1")(x)
    x = TransformerBlock(embed_dim, num_heads, ff_dim, name="transformer_2")(x)
    x = TransformerBlock(embed_dim, num_heads, ff_dim, name="transformer_3")(x)

    x      = tf.keras.layers.GlobalAveragePooling1D(name="pooling")(x)
    x      = tf.keras.layers.Dropout(0.25)(x)
    shared = tf.keras.layers.Dense(256, activation="relu", name="shared")(x)
    shared = tf.keras.layers.Dropout(0.15)(shared)

    nic_out = tf.keras.layers.Dense(num_nic, activation="softmax", name="nic_out")(shared)
    div_out = tf.keras.layers.Dense(num_div, activation="softmax", name="div_out")(shared)

    return tf.keras.Model(inputs=inputs, outputs=[nic_out, div_out], name="Udyam_NIC_v2")


model = build_model(
    vocab_size=MAX_VOCAB,
    embed_dim=EMBED_DIM,
    seq_len=SEQ_LEN,
    num_heads=6,
    ff_dim=384,
    num_nic=num_nic,
    num_div=num_div,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
    loss={
        "nic_out": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        "div_out": tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
    },
    loss_weights={"nic_out": 0.85, "div_out": 0.15},
    metrics={"nic_out": "accuracy", "div_out": "accuracy"},
)

print(f"\nTotal parameters: {model.count_params():,}")

# =====================================================
# 7. TRAIN
# =====================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_nic_out_accuracy", patience=7, verbose=1, mode="max",
        restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.4, patience=3, verbose=1, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.keras", monitor="val_nic_out_accuracy",
        save_best_only=True, verbose=1, mode="max"),
]

print("\n--- Training ---")
history = model.fit(
    X_train_vec,
    {"nic_out": yn_train, "div_out": yd_train},
    validation_data=(X_test_vec, {"nic_out": yn_test, "div_out": yd_test}),
    epochs=60,
    batch_size=64,
    callbacks=callbacks,
)

# =====================================================
# 8. EVALUATE
# =====================================================

results = model.evaluate(
    X_test_vec, {"nic_out": yn_test, "div_out": yd_test}, verbose=0)

print(f"\nNIC Code Accuracy  : {results[3] * 100:.1f}%")
print(f"Division Accuracy  : {results[4] * 100:.1f}%")

# =====================================================
# 9. SAVE
# =====================================================

model.save("models/udyam_nic_model.keras")
print("\nModel saved → models/udyam_nic_model.keras")
print("Vocab saved → models/vectorizer_vocab.json")
print("Labels saved → models/label_map.json")