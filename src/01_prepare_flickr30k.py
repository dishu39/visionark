import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image
import pandas as pd
from tqdm import tqdm

# -----------------------------------
# Config
# -----------------------------------
DATA_ROOT = Path("data")
IMAGES_DIR = DATA_ROOT / "raw" / "images" / "flickr30k"
OUTPUT_CSV = DATA_ROOT / "processed" / "flickr30k_pairs.csv"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)


def extract_captions(example):
    """
    Try to extract a list of captions from a single dataset example.
    Different HF flickr30k variants sometimes use slightly different keys.
    Adjust here if needed after printing an example once.
    """
    # Common possibilities:
    # - example["sentences"] is a list of dicts with "raw" field
    # - example["captions"] is a list of strings
    # - example["caption"] is a single string
    captions = []

    if "sentences" in example:
        # often: list of {'raw': 'caption text', ...}
        for s in example["sentences"]:
            if isinstance(s, dict) and "raw" in s:
                captions.append(s["raw"])
            elif isinstance(s, str):
                captions.append(s)
    elif "captions" in example:
        if isinstance(example["captions"], list):
            captions.extend(example["captions"])
        else:
            captions.append(str(example["captions"]))
    elif "caption" in example:
        if isinstance(example["caption"], list):
            captions.extend(example["caption"])
        else:
            captions.append(str(example["caption"]))
    else:
        # Fallback: try "text"
        if "text" in example:
            if isinstance(example["text"], list):
                captions.extend(example["text"])
            else:
                captions.append(str(example["text"]))

    # remove empties & strip
    captions = [c.strip() for c in captions if isinstance(c, str) and c.strip()]
    return captions


def get_image_id(example, idx):
    """
    Get a stable image id string for naming files and indexing.
    """
    for key in ["image_id", "id", "imgid"]:
        if key in example:
            return str(example[key])
    # Fallback: use dataset index
    return f"{idx:06d}"


def save_image(example, image_id):
    """
    Save the PIL image to disk and return its relative path as string.
    """
    # HF datasets usually store image as example["image"] (PIL.Image)
    img = example.get("image", None)
    if img is None:
        raise ValueError("No 'image' field found in example; inspect dataset schema.")

    if not isinstance(img, Image.Image):
        # some datasets store as array, convert
        img = Image.fromarray(img)

    filename = f"flickr30k_{image_id}.jpg"
    out_path = IMAGES_DIR / filename
    img.save(out_path, format="JPEG")
    return str(out_path)


def main():
    print("Loading Flickr30k from Hugging Face (nlphuji/flickr30k)...")
    ds = load_dataset("nlphuji/flickr30k")  # usually only 'train' split
    split_name = "train"
    dtrain = ds[split_name]

    print("Inspecting first example to verify keys...")
    first_example = dtrain[0]
    print("Example keys:", first_example.keys())
    # You can uncomment this to see structure once:
    # print(first_example)

    records = []

    print("Processing dataset and saving images + captions...")
    for idx, example in tqdm(enumerate(dtrain), total=len(dtrain)):
        image_id = get_image_id(example, idx)
        try:
            image_path = save_image(example, image_id)
        except Exception as e:
            print(f"[WARN] Skipping index {idx} due to image save error: {e}")
            continue

        captions = extract_captions(example)
        if not captions:
            # no valid captions, skip
            continue

        # For each caption create a row → creates multiple pairs per image
        for cap in captions:
            records.append(
                {
                    "image_id": image_id,
                    "image_path": image_path,
                    "caption": cap,
                    "split": split_name,
                }
            )

    if not records:
        print("No records collected. Check extract_captions() and dataset schema.")
        return

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} image-text pairs to {OUTPUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
