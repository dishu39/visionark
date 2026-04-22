import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import argparse
import os
import sys
import tkinter as tk
from tkinter import filedialog
import collections
import numpy as np
import csv
import random
from typing import Optional

# Optional text-to-speech (TTS)
try:
    import pyttsx3
    _HAS_TTS = True
except Exception:
    pyttsx3 = None
    _HAS_TTS = False

# try to import cv2 for better denoising
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# =========================================================
# CONFIG
# =========================================================
# Points to the model you just trained in Step 1
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_final_noise_robust")
if not os.path.exists(MODEL_DIR):
    # Fallback to standard name if robust version doesn't exist
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_final")

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from: {MODEL_DIR}")
processor = BlipProcessor.from_pretrained(MODEL_DIR)
model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

# =========================================================
# IMAGE CLEANING (DEEP CLEAN LOGIC)
# =========================================================

def denoise_image(pil_img, method="nlm", strength="aggressive"):
    """
    Advanced denoising. For 'aggressive' noise:
    1. Median Filter (instantly deletes black/white dots)
    2. NLM Denoising (smooths gaussian grain)
    3. Sharpening (restores edges)
    """
    if method is None or method.lower() == "none":
        return pil_img
    
    # Stage 1: Median Blur is the BEST defense against Salt & Pepper noise
    if strength in ("aggressive", "strong"):
        pil_img = pil_img.filter(ImageFilter.MedianFilter(size=5))

    if _HAS_CV2 and method.lower() in ("nlm", "bilateral"):
        arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if method.lower() == "nlm":
            h = 30 if strength == "aggressive" else 15
            den = cv2.fastNlMeansDenoisingColored(arr, None, h, h, 7, 21)
            # Stage 3: Recover object boundaries
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            den = cv2.filter2D(den, -1, kernel)
        else: # bilateral
            d = 15 if strength == "aggressive" else 9
            den = cv2.bilateralFilter(arr, d, 100, 100)
        
        return Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))
    
    return pil_img

def enhance_image(pil_img, method="clahe"):
    if not _HAS_CV2: return pil_img
    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    if method == "clahe":
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        lab[:, :, 0] = clahe
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    return pil_img

def detect_noise_level(pil_img):
    if not _HAS_CV2: return "medium"
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    variances = [np.var(arr[y:y+20, x:x+20]) for _ in range(20) 
                 for y, x in [(np.random.randint(0, h-20), np.random.randint(0, w-20))]]
    avg_var = np.mean(variances)
    if avg_var > 3500: return "aggressive"
    if avg_var > 1500: return "strong"
    return "light"

# =========================================================
# CAPTION SCORING (ARTIFACT-AWARE)
# =========================================================

def score_caption(caption):
    cap = caption.lower()
    words = cap.split()
    
    # 1. HEAVY PENALTY for describing the noise instead of objects
    noise_artifacts = ["dots", "multicolored", "static", "pixelated", "grainy", "spots", "blurry", "noise"]
    noise_penalty = sum(10 for word in noise_artifacts if word in cap)
    
    # 2. REWARD for descriptive objects and actions
    rewards = ["man", "woman", "person", "people", "holding", "standing", "sitting", 
               "wearing", "balloons", "shirt", "background", "colorful", "blue", "red"]
    desc_score = sum(5 for word in rewards if word in cap)
    
    # 3. Reward length (prevents 2-word captions)
    length_bonus = len(words) * 0.5
    
    return desc_score + length_bonus - noise_penalty

# =========================================================
# CORE GENERATION LOGIC
# =========================================================

def generate_caption(image_path, denoise=None, num_beams=5, max_tokens=60, temperature=1.0, tta=True):
    image = Image.open(image_path).convert("RGB")
    noise_lvl = detect_noise_level(image)
    print(f"--- Detected Noise Level: {noise_lvl} ---")

    # Determine variants for Test-Time Augmentation
    variants = {"original": image}
    if tta:
        variants["cleaned_nlm"] = denoise_image(image, method="nlm", strength=noise_lvl)
        variants["cleaned_enhanced"] = enhance_image(variants["cleaned_nlm"], method="clahe")
        if noise_lvl in ("strong", "aggressive"):
            variants["median_only"] = image.filter(ImageFilter.MedianFilter(size=7))

    all_candidates = []
    prompts = ["a photo of", "a scene showing", ""]

    print("Generating candidates...")
    
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "num_beams": num_beams,
        "length_penalty": 1.2, # Encourages detail
        "repetition_penalty": 1.5,
        "no_repeat_ngram_size": 3
    }
    if temperature != 1.0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["do_sample"] = True

    for name, img in variants.items():
        for pr in prompts:
            inputs = processor(images=img, text=pr, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
            cap = processor.decode(outputs[0], skip_special_tokens=True).strip()
            if cap: all_candidates.append(cap)

    # Use Artifact-Aware Scorer to pick the best result
    scored = [(c, score_caption(c)) for c in set(all_candidates)]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_caption = scored[0][0] if scored else "Could not generate caption."
    
    print("\nTop 3 Candidates:")
    for c, s in scored[:3]:
        print(f" - (Score: {s:.1f}) {c}")

    return best_caption, variants.get("cleaned_enhanced", image)

def speak_caption(caption):
    if _HAS_TTS and caption:
        engine = pyttsx3.init()
        engine.setProperty("rate", 160)
        engine.say(caption)
        engine.runAndWait()

# =========================================================
# MAIN EXECUTION
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?", help="Path to image")
    parser.add_argument("--tta", action="store_true", default=True)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    img_p = args.image
    if not img_p:
        root = tk.Tk(); root.withdraw()
        img_p = filedialog.askopenfilename()
        root.destroy()

    if not img_p or not os.path.exists(img_p):
        print("No valid image selected."); sys.exit(1)

    caption, cleaned_img = generate_caption(img_p, num_beams=args.num_beams, max_tokens=args.max_tokens, temperature=args.temperature, tta=args.tta)

    print(f"\nFINAL CAPTION: {caption}")
    speak_caption(caption)

    # Display Result
    plt.figure(figsize=(10, 7))
    plt.imshow(cleaned_img)
    plt.title(f"Generated Caption:\n{caption}", fontsize=12)
    plt.axis("off")
    plt.show()