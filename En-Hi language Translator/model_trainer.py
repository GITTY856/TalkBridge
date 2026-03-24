import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate

# ============================================================
#  LANGUAGE CONFIGURATION — CHANGE HERE WHEN SWITCHING DIRECTION
# ============================================================

# For Hindi → English:
MODEL_NAME      = "Helsinki-NLP/opus-mt-hi-en"   # ← CHANGE: "opus-mt-en-hi" for En→Hi
SOURCE_LANG     = "hi"                            # ← CHANGE: "en" for En→Hi
TARGET_LANG     = "en"                            # ← CHANGE: "hi" for En→Hi
SOURCE_COL      = "Hindi"                         # ← CHANGE: "English" for En→Hi
TARGET_COL      = "English"                       # ← CHANGE: "Hindi" for En→Hi
OUTPUT_DIR_NAME = "my_en_translator"              # ← CHANGE: "my_hi_translator" for En→Hi

# ============================================================
#  ENVIRONMENT DETECTION
# ============================================================

IN_COLAB = os.path.exists("/content")

print("=" * 60)
print(f"  Environment : {'Google Colab' if IN_COLAB else 'Local Machine'}")
print(f"  CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)} GB")
else:
    print("  WARNING: No GPU found — training will be slow!")
print(f"  Direction : {SOURCE_LANG.upper()} → {TARGET_LANG.upper()}")
print("=" * 60)

# ============================================================
#  GOOGLE DRIVE MOUNT (Colab only)
# ============================================================

if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        print("[Colab] Google Drive mounted.")
    except Exception as e:
        print(f"[Colab] Drive mount skipped: {e}")

# ============================================================
#  PATH CONFIGURATION
# ============================================================

if IN_COLAB and os.path.exists("/content/drive/MyDrive"):
    # Colab + Google Drive — all files saved permanently
    DRIVE_BASE   = "/content/drive/MyDrive/Translator"
    CSV_PATH     = f"{DRIVE_BASE}/Cleaned_Dataset_Final.csv"
    OUTPUT_DIR   = f"{DRIVE_BASE}/{OUTPUT_DIR_NAME}"
elif IN_COLAB:
    # Colab without Drive — files lost on session end
    CSV_PATH   = "/content/Cleaned_Dataset_Final.csv"
    OUTPUT_DIR = f"/content/{OUTPUT_DIR_NAME}"
else:
    # Local machine
    CSV_PATH   = r"C:\Langauge Translator Project\En-Hi language Translator\Cleaned_Dataset_Final.csv"
    OUTPUT_DIR = OUTPUT_DIR_NAME

print(f"[Paths] CSV      : {CSV_PATH}")
print(f"[Paths] Output   : {OUTPUT_DIR}")

# ============================================================
#  BATCH SIZE — AUTO DETECTED FROM VRAM
# ============================================================

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if vram_gb >= 15:        # Colab T4 / A100
        BATCH_SIZE = 32
        GRAD_ACCUM = 1       # effective batch = 32
    else:                    # Local RTX 4050 (6 GB)
        BATCH_SIZE = 16
        GRAD_ACCUM = 2       # effective batch = 32
else:
    BATCH_SIZE = 4           # CPU fallback
    GRAD_ACCUM = 8           # effective batch = 32

print(f"[Batch] per_device={BATCH_SIZE}  grad_accum={GRAD_ACCUM}  effective={BATCH_SIZE * GRAD_ACCUM}")

# ============================================================
#  LOAD DATASET
# ============================================================

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"\n[ERROR] Dataset not found at: {CSV_PATH}\n"
        "  Colab users: Upload CSV to Google Drive or use:\n"
        "    from google.colab import files; files.upload()\n"
        "  Then update CSV_PATH above."
    )

df = pd.read_csv(CSV_PATH)
df = df[[SOURCE_COL, TARGET_COL]].dropna()
df.columns = [SOURCE_LANG, TARGET_LANG]

print(f"[Data] Total samples : {len(df):,}")

# Train / val split
train_size   = int(0.95 * len(df))
train_df     = df[:train_size]
val_df       = df[train_size:]
train_dataset = Dataset.from_pandas(train_df)
val_dataset   = Dataset.from_pandas(val_df)

print(f"[Data] Train : {len(train_df):,}   Val : {len(val_df):,}")

# ============================================================
#  LOAD MODEL & TOKENIZER
# ============================================================

print(f"\n[Model] Loading {MODEL_NAME} ...")
tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model     = MarianMTModel.from_pretrained(MODEL_NAME)

# ============================================================
#  TOKENIZATION
# ============================================================

def preprocess_function(examples):
    inputs  = examples[SOURCE_LANG]    # ← driven by SOURCE_LANG variable above
    targets = examples[TARGET_LANG]    # ← driven by TARGET_LANG variable above
    model_inputs = tokenizer(inputs,  max_length=128, truncation=True, padding=False)
    labels       = tokenizer(text_target=targets, max_length=128, truncation=True, padding=False)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

remove_cols   = [SOURCE_LANG, TARGET_LANG]
train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=remove_cols)
val_dataset   = val_dataset.map(preprocess_function,   batched=True, remove_columns=remove_cols)

# ============================================================
#  DATA COLLATOR
# ============================================================

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ============================================================
#  BLEU METRIC
# ============================================================

bleu_metric = evaluate.load("sacrebleu")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )
    return {"bleu": round(result["score"], 2)}

# ============================================================
#  TRAINING ARGUMENTS
# ============================================================

training_args = Seq2SeqTrainingArguments(
    output_dir                  = OUTPUT_DIR,
    eval_strategy               = "steps",
    eval_steps                  = 200,        # more frequent saves for Colab safety
    save_steps                  = 200,
    save_total_limit            = 3,          # keep 3 checkpoints (1 extra safety)
    learning_rate               = 5e-5,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    num_train_epochs            = 3,
    weight_decay                = 0.01,
    warmup_steps                = 200,
    fp16                        = torch.cuda.is_available(),
    logging_steps               = 100,
    predict_with_generate       = True,
    load_best_model_at_end      = True,
    metric_for_best_model       = "bleu",
    greater_is_better           = True,
)

# ============================================================
#  TRAINER
# ============================================================

trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    processing_class = tokenizer,     # updated from deprecated 'tokenizer=' param
    data_collator   = data_collator,
    compute_metrics = compute_metrics,
)

# ============================================================
#  RESUME FROM CHECKPOINT (safe for Colab disconnections)
# ============================================================

last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint")
    ]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"\n[Checkpoint] Resuming from: {last_checkpoint}")
    else:
        print("\n[Checkpoint] No checkpoint found — starting fresh.")
else:
    print("\n[Checkpoint] Starting fresh.")

# ============================================================
#  TRAIN
# ============================================================

print("\nStarting training...")
trainer.train(resume_from_checkpoint=last_checkpoint)

# ============================================================
#  SAVE FINAL MODEL
# ============================================================

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"\nTraining complete! Model saved to: {OUTPUT_DIR}")
