import os
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    Trainer,
    TrainingArguments
)


# =========================================================
# CONFIG
# =========================================================

# Update this path based on your folder structure
PROJECT_DIR = Path("D:\\B.Tech\\PROJECT\\root")

CSV_PATH = PROJECT_DIR / "D:\\B.Tech\\PROJECT\\root\\data\\processed\\flickr30k_clip_clean.csv"
MODEL_NAME = "Salesforce/blip-image-captioning-base"
SAVE_DIR = PROJECT_DIR / "model_final"
SUBSET_SIZE = 8000      # use subset for faster training (can increase)


# =========================================================
# STEP 1 - Load dataset
# =========================================================

print("\nLoading dataset:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

print("Original dataset size:", len(df))
df = df.sample(min(SUBSET_SIZE, len(df)), random_state=42).reset_index(drop=True)
print("Training subset size:", len(df))


# Fix image paths if needed (appends Google Drive prefix only if missing)
def fix_path(p: str) -> str:
    if p.startswith("data/"):
        return str(PROJECT_DIR / p)
    return p

df["image_path"] = df["image_path"].apply(fix_path)

# Test one file exists
sample_path = df["image_path"].iloc[0]
print("\nTesting image path:", sample_path, "Exists?", os.path.exists(sample_path))


# =========================================================
# STEP 2 - Create PyTorch Dataset class
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nUsing device:", device)

processor = BlipProcessor.from_pretrained(MODEL_NAME)

class FlickrDataset(Dataset):
    def __init__(self, dataframe, processor):
        self.df = dataframe
        self.processor = processor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        caption = str(row["caption"])

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))

        encoding = self.processor(
            images=image,
            text=caption,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = encoding["input_ids"].clone()

        return encoding


# Split train/val (90/10)
val_size = int(len(df) * 0.1)
train_df = df.iloc[:-val_size].reset_index(drop=True)
val_df = df.iloc[-val_size:].reset_index(drop=True)

train_dataset = FlickrDataset(train_df, processor)
val_dataset = FlickrDataset(val_df, processor)

print("\nDataset ready: train =", len(train_dataset), "| val =", len(val_dataset))


# =========================================================
# STEP 3 - Load BLIP Model
# =========================================================

print("\nLoading model:", MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)


# =========================================================
# STEP 4 - Training Setup
# =========================================================

def collate_fn(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    return out

training_args = TrainingArguments(
    output_dir=str(PROJECT_DIR / "training_logs"),
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    fp16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn
)

print("\n🚀 Starting Training...\n")
trainer.train()


# =========================================================
# STEP 5 - Save Final Model
# =========================================================

SAVE_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(SAVE_DIR)
processor.save_pretrained(SAVE_DIR)

print("\n✔️ Training complete!")
print("Model saved to:", SAVE_DIR)
