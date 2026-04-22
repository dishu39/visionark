import os
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
import open_clip

DATA_DIR = Path("data/processed")
INPUT_CSV = DATA_DIR / "flickr30k_lang_clean.csv"
OUTPUT_CSV = DATA_DIR / "flickr30k_clip_scored.csv"
OUTPUT_CLEAN_CSV = DATA_DIR / "flickr30k_clip_clean.csv"

HARD_THRESH = 0.15
SOFT_THRESH = 0.25
BATCH_SIZE = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

df = pd.read_csv(INPUT_CSV)
print("Total pairs before CLIP filtering:", len(df))
print("Sample path:", df["image_path"].iloc[0])

# if your image paths still have a Windows absolute prefix, fix them here
# Example (edit this string to match your actual local prefix if needed):
# df["image_path"] = df["image_path"].str.replace(
#     r"C:\\Users\\yourname\\path\\to\\project\\",
#     "/content/drive/MyDrive/vision_noise_project/",
#     regex=False,
# )

print("Check exists:", os.path.exists(df["image_path"].iloc[0]))

df.to_csv(INPUT_CSV, index=False)

print("Loading CLIP model...")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")
model = model.to(device)
model.eval()

def compute_clip_sims(image_paths, texts):
    images = []
    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[WARN] image load failed {p}: {e}")
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        images.append(preprocess(img))

    image_tensor = torch.stack(images).to(device)
    text_tokens = tokenizer(texts).to(device)

    with torch.no_grad():
        img_feat = model.encode_image(image_tensor)
        txt_feat = model.encode_text(text_tokens)
        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)
        sims = (img_feat * txt_feat).sum(dim=-1)

    return sims.cpu()

sims = []
keep_flags = []
buckets = []

n = len(df)
idx = 0

while idx < n:
    batch = df.iloc[idx: idx + BATCH_SIZE]
    paths = batch["image_path"].tolist()
    caps = batch["caption"].astype(str).tolist()

    batch_sims = compute_clip_sims(paths, caps)

    for s in batch_sims:
        v = float(s.item())
        sims.append(v)
        if v < HARD_THRESH:
            keep_flags.append(False); buckets.append("hard_drop")
        elif v < SOFT_THRESH:
            keep_flags.append(True); buckets.append("borderline")
        else:
            keep_flags.append(True); buckets.append("keep")

    idx += BATCH_SIZE
    if idx % (BATCH_SIZE * 20) == 0:
        print(f"Processed {idx}/{n}")

df["clip_sim"] = sims
df["keep_clip"] = keep_flags
df["clip_bucket"] = buckets

print("Mean CLIP similarity:", df["clip_sim"].mean())

df.to_csv(OUTPUT_CSV, index=False)
print("Saved scored pairs to:", OUTPUT_CSV)

df_clean = df[df["keep_clip"]].copy()
print("Pairs after CLIP filtering:", len(df_clean))

df_clean.to_csv(OUTPUT_CLEAN_CSV, index=False)
print("Saved CLIP-clean pairs to:", OUTPUT_CLEAN_CSV)
df_clean.head()
