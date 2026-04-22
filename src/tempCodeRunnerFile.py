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
# ...existing code...
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

# Metrics deps are optional; we import inside eval to keep normal inference working.

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model_final")

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained(MODEL_DIR)
model = BlipForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()

# try to import cv2 for better denoising, otherwise fallback to PIL
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

def denoise_image(pil_img, method="nlm", strength="medium"):
    """
    Denoise image with various methods.
    strength: "light", "medium", "strong", "aggressive"
    """
    if method is None or method.lower() == "none":
        return pil_img
    
    if _HAS_CV2 and method.lower() in ("nlm", "bilateral"):
        arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if method.lower() == "nlm":
            # Adjust parameters based on strength
            if strength == "light":
                h, hColor, templateWindowSize, searchWindowSize = 10, 10, 7, 21
            elif strength == "medium":
                h, hColor, templateWindowSize, searchWindowSize = 15, 15, 7, 21
            elif strength == "strong":
                h, hColor, templateWindowSize, searchWindowSize = 20, 20, 7, 21
            else:  # aggressive
                h, hColor, templateWindowSize, searchWindowSize = 30, 30, 7, 21
            
            den = cv2.fastNlMeansDenoisingColored(arr, None, h, hColor, templateWindowSize, searchWindowSize)
            
            # For aggressive, apply a second pass
            if strength == "aggressive":
                den = cv2.fastNlMeansDenoisingColored(den, None, 15, 15, 7, 21)
        else:  # bilateral
            if strength == "light":
                d, sigmaColor, sigmaSpace = 5, 50, 50
            elif strength == "medium":
                d, sigmaColor, sigmaSpace = 9, 75, 75
            elif strength == "strong":
                d, sigmaColor, sigmaSpace = 11, 100, 100
            else:  # aggressive
                d, sigmaColor, sigmaSpace = 15, 150, 150
            
            den = cv2.bilateralFilter(arr, d, sigmaColor, sigmaSpace)
        
        den = cv2.cvtColor(den, cv2.COLOR_BGR2RGB)
        return Image.fromarray(den)
    
    # fallback to simple PIL filters
    if method.lower() == "median":
        size = 3 if strength in ("light", "medium") else 5
        return pil_img.filter(ImageFilter.MedianFilter(size=size))
    if method.lower() == "sharpen":
        return pil_img.filter(ImageFilter.SHARPEN)
    return pil_img

def enhance_image(pil_img, method="clahe"):
    """Apply image enhancement techniques."""
    if not _HAS_CV2:
        return pil_img
    
    arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    if method == "clahe":
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        lab[:, :, 0] = clahe
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif method == "histeq":
        # Histogram equalization
        yuv = cv2.cvtColor(arr, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    else:
        enhanced = arr
    
    return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))

def detect_noise_level(pil_img):
    """Simple noise detection based on variance in small patches."""
    if not _HAS_CV2:
        return "medium"
    
    arr = np.array(pil_img)
    # Sample random patches and compute variance
    h, w = arr.shape[:2]
    patches = []
    for _ in range(20):  # More samples for better detection
        y = np.random.randint(0, max(1, h - 20))
        x = np.random.randint(0, max(1, w - 20))
        patch = arr[y:y+20, x:x+20]
        patches.append(np.var(patch))
    
    avg_variance = np.mean(patches)
    max_variance = np.max(patches)
    
    # More aggressive thresholds for heavily pixelated images
    if avg_variance > 4000 or max_variance > 8000:
        return "aggressive"
    elif avg_variance > 2500 or max_variance > 5000:
        return "strong"
    elif avg_variance > 1200:
        return "medium"
    else:
        return "light"

def make_variants(pil_img, auto_detect_noise=True, fast_mode=False):
    """Create enhanced variants for test-time augmentation (TTA)."""
    variants = {}
    
    # Auto-detect noise level if requested
    noise_level = detect_noise_level(pil_img) if auto_detect_noise else "medium"
    print(f"Detected noise level: {noise_level}")
    
    if fast_mode:
        # Fast mode: keep small set; if very noisy, add median/bilateral for robustness
        variants["orig"] = pil_img
        if _HAS_CV2:
            variants["nlm_strong"] = denoise_image(pil_img, method="nlm", strength="strong")
            denoised = denoise_image(pil_img, method="nlm", strength=noise_level)
            variants["nlm_clahe"] = enhance_image(denoised, method="clahe")
            if noise_level in ("strong", "aggressive"):
                variants["bilateral"] = denoise_image(pil_img, method="bilateral", strength="medium")
        else:
            variants["median"] = pil_img.filter(ImageFilter.MedianFilter(size=5))
        return variants
    
    if _HAS_CV2:
        # For very noisy images, skip original and focus on heavily processed versions
        if noise_level not in ("aggressive", "strong"):
            variants["orig"] = pil_img
        
        # Key denoising variants (reduced from 10+ to 5-6)
        variants["nlm_strong"] = denoise_image(pil_img, method="nlm", strength="strong")
        variants["nlm_aggressive"] = denoise_image(pil_img, method="nlm", strength="aggressive")
        
        # Multi-pass denoising (only if very noisy)
        if noise_level in ("strong", "aggressive"):
            temp1 = denoise_image(pil_img, method="nlm", strength="aggressive")
            variants["nlm_triple"] = denoise_image(temp1, method="nlm", strength="strong")
        
        # Combined: aggressive denoise + enhance (most effective)
        denoised_aggressive = denoise_image(pil_img, method="nlm", strength="aggressive")
        variants["nlm_aggressive_clahe"] = enhance_image(denoised_aggressive, method="clahe")
        
        # One medium variant for comparison
        variants["nlm_medium"] = denoise_image(pil_img, method="nlm", strength="medium")
    else:
        # Fallback PIL filters
        variants["median_large"] = pil_img.filter(ImageFilter.MedianFilter(size=5))
    
    return variants


def preprocess_image_for_noise(pil_img, noise_level):
    """
    Apply a single, strong pre-denoise step when noise is high.
    """
    if noise_level not in ("strong", "aggressive"):
        return pil_img

    if _HAS_CV2:
        den = denoise_image(pil_img, method="nlm", strength="aggressive")
        den = enhance_image(den, method="clahe")
        return den

    # Fallback without OpenCV: heavier median blur
    return pil_img.filter(ImageFilter.MedianFilter(size=7))


_tts_engine = None


def speak_caption(caption: str):
    """
    Speak the generated caption aloud using the system TTS engine (if available).
    """
    if not _HAS_TTS or not caption:
        if not _HAS_TTS:
            print("Text-to-speech dependency 'pyttsx3' not available; skipping voice output.")
        return

    global _tts_engine
    try:
        if _tts_engine is None:
            _tts_engine = pyttsx3.init()
            # Slightly slower than default for clarity
            _tts_engine.setProperty("rate", 175)
        _tts_engine.say(caption)
        _tts_engine.runAndWait()
    except Exception as e:
        print(f"Error during text-to-speech: {e}")

def generate_caption(image_path, denoise=None, max_new_tokens=60, num_beams=5, tta=True, 
                     temperature=0.9, length_penalty=1.5, auto_detect_noise=True, 
                     num_candidates=3, prompt=None, fast_mode=False):
    """
    Generate caption with improved handling for noisy images.
    
    Args:
        image_path: Path to image
        denoise: Explicit denoising method (overrides auto-detection if set)
        max_new_tokens: Maximum tokens to generate (increased for better descriptions)
        num_beams: Beam search width
        tta: Enable test-time augmentation
        temperature: Sampling temperature (lower = more deterministic)
        length_penalty: Length penalty (>1 encourages longer captions)
        auto_detect_noise: Automatically detect noise level and adjust preprocessing
        num_candidates: Number of candidate captions to generate per variant
        prompt: Optional prompt prefix to guide generation
    """
    # Score captions: prefer longer, more descriptive captions, penalize generic ones
    def score_caption(caption):
        caption_lower = caption.lower()
        words = caption.split()
        
        length_score = len(words)
        
        # Heavy penalty for very short or generic captions
        if length_score < 4:
            return -10
        if length_score < 6:
            return length_score * 0.3
        
        # Penalize generic/overly simple captions
        generic_phrases = ["a multicolored image", "a photo", "an image", "a picture", 
                          "a person's face", "a person", "a face", "multicolored image"]
        generic_penalty = sum(3 for phrase in generic_phrases if phrase in caption_lower)
        
        # Bonus for descriptive words - general vocabulary
        descriptive_words = [
            # People
            "person", "people", "man", "woman", "men", "women", "character", "characters",
            # Objects
            "balloon", "balloons", "laptop", "screen", "display", "monitor", "computer",
            # Actions
            "holding", "standing", "sitting", "wearing", "showing", "displaying", "featuring",
            # Descriptors
            "colorful", "multicolored", "two", "several", "anime", "cartoon", "animated",
            # Colors
            "red", "blue", "green", "yellow", "bright", "large", "white", "black",
            # Screen/display related
            "on screen", "on the screen", "screen showing", "displaying", "shows"
        ]
        desc_score = sum(2 for word in descriptive_words if word in caption_lower)
        
        # Special bonuses for specific content types
        content_bonus = 0
        # Screen/display content
        if any(word in caption_lower for word in ["screen", "laptop", "monitor", "display", "computer"]):
            content_bonus += 5
            if "showing" in caption_lower or "displaying" in caption_lower or "shows" in caption_lower:
                content_bonus += 5
        
        # Anime/cartoon content
        if "anime" in caption_lower or "cartoon" in caption_lower or "animated" in caption_lower:
            content_bonus += 5
            if "character" in caption_lower:
                content_bonus += 3
        
        # People with actions (general, not balloon-specific)
        has_people = any(word in caption_lower for word in ["men", "women", "man", "woman", "people", "person", "character"])
        has_action = any(word in caption_lower for word in ["holding", "standing", "sitting", "wearing", "showing"])
        
        if has_people and has_action:
            content_bonus += 5
        
        # Bonus for mentioning multiple objects/people/characters
        if any(word in caption_lower for word in ["two", "several", "multiple", "both"]):
            content_bonus += 3
        
        # Action words bonus
        action_words = ["holding", "standing", "sitting", "wearing", "showing", "displaying", "featuring"]
        action_score = sum(2 for word in action_words if word in caption_lower)
        
        # Structure bonus: longer, more detailed captions
        if length_score >= 8:
            structure_bonus = 3
        elif length_score >= 6:
            structure_bonus = 1
        else:
            structure_bonus = 0
        
        final_score = length_score + desc_score + action_score + content_bonus + structure_bonus - generic_penalty
        return final_score
    
    image = Image.open(image_path).convert("RGB")
    
    # if explicit denoise requested, apply single method
    if denoise and denoise != "none":
        noise_level = "strong" if denoise in ("nlm", "bilateral") else "medium"
        image = denoise_image(image, method=denoise, strength=noise_level)
    
    # Lightweight auto handling for very noisy images even when TTA is off
    noise_level_auto = None
    if auto_detect_noise and not denoise:
        noise_level_auto = detect_noise_level(image)
        print(f"Auto-detected noise level for generation: {noise_level_auto}")

    # Preprocess once for high-noise inputs
    base_image = preprocess_image_for_noise(image, noise_level_auto) if noise_level_auto else image

    # Use prompts to guide generation for better descriptions
    if prompt is None:
        if fast_mode:
            # Fast mode: only 2-3 key prompts
            prompts = [
                "a screen showing",
                "a photo of",
                ""
            ]
        else:
            # Diverse prompts that work for various image types (reduced from 9 to 5)
            prompts = [
                # Screen/display prompts
                "a laptop screen showing",
                "a screen showing",
                # General descriptive prompts
                "a photo of",
                "a picture showing",
                # No prompt (default behavior)
                ""
            ]
    else:
        prompts = [prompt]

    if not tta:
        # For very noisy images, run a lightweight variant-based search for robustness
        if noise_level_auto in ("strong", "aggressive"):
            variants = make_variants(base_image, auto_detect_noise=False, fast_mode=True)
            captions = []
            print(f"Using lightweight variants for noisy image ({noise_level_auto})...")
            for name, var_img in variants.items():
                try:
                    for prompt_text in prompts:
                        inputs = processor(images=var_img, text=prompt_text, return_tensors="pt", truncation=True).to(device)
                        gen_kwargs = {
                            "max_new_tokens": max(max_new_tokens, 45),  # allow a bit more room for detail on noisy inputs
                            "temperature": temperature,
                            "length_penalty": length_penalty,
                            "num_return_sequences": 3,  # a tad more diversity for noisy cases
                            "do_sample": True,
                            "top_p": 0.9,
                            "top_k": 50,
                        }
                        # Slight beam for very noisy images but keep small to stay fast
                        if num_beams and int(num_beams) > 1:
                            gen_kwargs["num_beams"] = min(3, int(num_beams))
                            gen_kwargs["no_repeat_ngram_size"] = 3
                            gen_kwargs["early_stopping"] = True
                            gen_kwargs["num_return_sequences"] = 1
                            gen_kwargs["do_sample"] = False
                        elif num_beams in (None, 0, 1):
                            # Force a tiny beam for very noisy inputs to stabilize without big cost
                            gen_kwargs["num_beams"] = 2
                            gen_kwargs["no_repeat_ngram_size"] = 3
                            gen_kwargs["early_stopping"] = True
                            gen_kwargs["num_return_sequences"] = 1
                            gen_kwargs["do_sample"] = False

                        with torch.no_grad():
                            out = model.generate(**inputs, **gen_kwargs)
                        for seq in out:
                            cap = processor.decode(seq, skip_special_tokens=True)
                            if prompt_text:
                                prompt_lower = prompt_text.strip().lower()
                                cap_lower = cap.lower()
                                if cap_lower.startswith(prompt_lower):
                                    cap = cap[len(prompt_text.strip()):].strip()
                                cap = cap.replace(prompt_text.strip(), "").strip()
                                cap = " ".join(cap.split())
                            if cap:
                                captions.append((name, cap))
                except Exception as e:
                    print(f"  {name}: Error in noisy-image variant generation - {e}")
                    continue

            if captions:
                scored = [(cap, score_caption(cap)) for _, cap in captions]
                scored.sort(key=lambda x: x[1], reverse=True)
                return scored[0][0], image

        # Normal single-image generation (fast path)
        all_captions = []
        for prompt_text in prompts:
            inputs = processor(images=base_image, text=prompt_text, return_tensors="pt", truncation=True).to(device)
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "length_penalty": length_penalty,
                "num_return_sequences": num_candidates,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50
            }
            if num_beams and int(num_beams) > 1:
                gen_kwargs["num_beams"] = int(num_beams)
                gen_kwargs["no_repeat_ngram_size"] = 3
                gen_kwargs["early_stopping"] = True
                gen_kwargs["num_return_sequences"] = 1
                gen_kwargs["do_sample"] = False
            
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            for seq in out:
                cap = processor.decode(seq, skip_special_tokens=True)
                if prompt_text:
                    # Remove prompt more intelligently
                    prompt_lower = prompt_text.strip().lower()
                    cap_lower = cap.lower()
                    if cap_lower.startswith(prompt_lower):
                        cap = cap[len(prompt_text.strip()):].strip()
                    # Also try removing if it appears anywhere
                    cap = cap.replace(prompt_text.strip(), "").strip()
                    # Clean up any double spaces
                    cap = " ".join(cap.split())
                all_captions.append(("single", cap))
        
        # Score and return best
        scored = [(cap, score_caption(cap)) for _, cap in all_captions]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0], base_image

    # TTA: generate on several enhanced variants and pick the best caption
    variants = make_variants(base_image, auto_detect_noise=auto_detect_noise, fast_mode=fast_mode)
    captions = []
    
    # Adjust candidates for fast mode
    actual_candidates = 2 if fast_mode else num_candidates
    
    print(f"Generating captions on {len(variants)} variants with {actual_candidates} candidates each...")
    for name, var_img in variants.items():
        try:
            # Generate multiple candidates per variant with different prompts
            variant_captions = []
            for prompt_text in prompts:
                inputs = processor(images=var_img, text=prompt_text, return_tensors="pt", truncation=True).to(device)
                
                # Use either beam search OR sampling, not both (for speed)
                if num_beams and int(num_beams) > 1 and not fast_mode:
                    # Use beam search (more deterministic, faster)
                    gen_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "length_penalty": length_penalty,
                        "num_beams": int(num_beams),
                        "no_repeat_ngram_size": 3,
                        "early_stopping": True,
                        "num_return_sequences": min(2, actual_candidates),  # Limit beam search results
                        "do_sample": False
                    }
                else:
                    # Use sampling (more diverse)
                    gen_kwargs = {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "length_penalty": length_penalty,
                        "num_return_sequences": actual_candidates,
                        "do_sample": True,
                        "top_p": 0.9,
                        "top_k": 50
                    }
                
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                for seq in out:
                    cap = processor.decode(seq, skip_special_tokens=True)
                    if prompt_text:
                        # Remove prompt more intelligently
                        prompt_lower = prompt_text.strip().lower()
                        cap_lower = cap.lower()
                        if cap_lower.startswith(prompt_lower):
                            cap = cap[len(prompt_text.strip()):].strip()
                        cap = cap.replace(prompt_text.strip(), "").strip()
                        cap = " ".join(cap.split())
                    variant_captions.append(cap)
            
            # Add all unique captions from this variant
            for cap in variant_captions:
                if cap and len(cap.strip()) > 0:
                    captions.append((name, cap))
            
            if variant_captions:
                print(f"  {name}: {variant_captions[0][:70]}...")
        except Exception as e:
            print(f"  {name}: Error - {e}")
            continue

    if not captions:
        # Fallback to single generation
        inputs = processor(images=image, return_tensors="pt").to(device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "length_penalty": length_penalty,
            "num_return_sequences": 1
        }
        if num_beams and int(num_beams) > 1:
            gen_kwargs["num_beams"] = int(num_beams)
            gen_kwargs["no_repeat_ngram_size"] = 3
            gen_kwargs["early_stopping"] = True
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        return processor.decode(out[0], skip_special_tokens=True), image

    # Score all captions
    scored = [(cap, score_caption(cap)) for _, cap in captions]
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Get top candidates
    top_n = min(10, len(scored))
    top_captions = [cap for cap, score in scored[:top_n]]
    
    # Prioritize captions that have ideal formats for different content types
    ideal_patterns = [
        # Screen/display patterns
        ("screen showing", ["laptop", "computer", "monitor", "display"]),
        ("showing", ["screen", "laptop", "anime", "character"]),
        ("displaying", ["screen", "laptop", "monitor"]),
        # People/character patterns
        ("men and women holding", ["balloon", "balloons"]),
        ("people holding", ["balloon", "balloons"]),
        ("characters", ["anime", "cartoon", "showing", "holding"]),
        # General descriptive patterns
        ("a photo of", ["people", "characters", "scene"]),
        ("a picture showing", ["people", "characters", "scene"])
    ]
    
    # Check if any top captions match ideal patterns
    ideal_matches = []
    for cap, score in scored[:top_n]:
        cap_lower = cap.lower()
        for pattern, context_words in ideal_patterns:
            if pattern in cap_lower:
                # Check if context words are present (at least one)
                if any(word in cap_lower for word in context_words):
                    ideal_matches.append((cap, score + 10))  # Extra boost
                    break
    
    if ideal_matches:
        # Sort ideal matches by score
        ideal_matches.sort(key=lambda x: x[1], reverse=True)
        best_caption = ideal_matches[0][0]
        print(f"\nFound ideal format caption!")
    else:
        # Also check most common among top-scored
        top_counts = collections.Counter(top_captions)
        if top_counts:
            most_common_top = top_counts.most_common(1)[0][0]
            # If most common is also high-scoring, prefer it
            most_common_score = next(score for cap, score in scored if cap == most_common_top)
            if most_common_score >= scored[0][1] * 0.8:  # Within 20% of best
                best_caption = most_common_top
            else:
                best_caption = scored[0][0]
        else:
            best_caption = scored[0][0]
    
    print(f"\nTop 5 candidates:")
    for i, (cap, score) in enumerate(scored[:5], 1):
        cap_lower = cap.lower()
        # Check if this caption matches any ideal pattern
        has_ideal = False
        for pattern, context_words in ideal_patterns:
            if pattern in cap_lower and any(word in cap_lower for word in context_words):
                has_ideal = True
                break
        marker = " ⭐" if has_ideal else ""
        print(f"  {i}. (score: {score:.1f}) {cap}{marker}")
    print(f"\nSelected caption: {best_caption}")
    return best_caption, image

# -----------------------------
# Evaluation (caption metrics)
# -----------------------------
def _group_references_from_pairs_csv(pairs_csv_path: str, split: str = "test", limit_images: int | None = None,
                                    seed: int = 42):
    """
    Returns:
      image_paths: List[str] length N
      references: List[List[str]] length N, each inner list contains all reference captions for that image
    """
    split_norm = str(split).lower()
    refs_by_path: dict[str, list[str]] = {}

    with open(pairs_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"image_path", "caption", "split"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {pairs_csv_path}: {sorted(missing)}")

        for row in reader:
            if str(row.get("split", "")).lower() != split_norm:
                continue
            img_path = row.get("image_path", "")
            cap = row.get("caption", "")
            if not img_path or not cap:
                continue
            refs_by_path.setdefault(img_path, []).append(cap)

    if not refs_by_path:
        raise ValueError(f"No rows found for split='{split}' in {pairs_csv_path}")

    # Reproducible sampling over images
    image_paths = list(refs_by_path.keys())
    rng = random.Random(seed)
    rng.shuffle(image_paths)
    if limit_images is not None:
        image_paths = image_paths[: int(limit_images)]

    references = [refs_by_path[p] for p in image_paths]
    return image_paths, references


def evaluate_on_flickr30k_pairs(
    pairs_csv_path: str,
    split: str = "test",
    limit_images: int | None = 500,
    seed: int = 42,
    denoise=None,
    max_new_tokens: int = 60,
    num_beams: int = 5,
    tta: bool = False,
    temperature: float = 0.9,
    length_penalty: float = 1.5,
    auto_detect_noise: bool = True,
    num_candidates: int = 1,
    prompt=None,
    fast_mode: bool = True,
):
    """
    Computes captioning metrics on Flickr30k:
      - BLEU via sacreBLEU (corpus_bleu, supports multiple references)
      - ROUGE-1/2/L via rouge-score (we take best-over-references per sample)
      - METEOR via NLTK (best-over-references per sample)

    This avoids `evaluate` which can be brittle on bleeding-edge Python versions.
    """
    image_paths, references = _group_references_from_pairs_csv(
        pairs_csv_path, split=split, limit_images=limit_images, seed=seed
    )

    predictions = []
    used_image_paths = []
    missing = 0
    for i, img_path in enumerate(image_paths, 1):
        if i == 1 or i % 25 == 0 or i == len(image_paths):
            print(f"Processed {i-1}/{len(image_paths)} images...", flush=True)
        if not os.path.isfile(img_path):
            # Skip missing images rather than failing the whole eval
            missing += 1
            continue
        cap, _ = generate_caption(
            img_path,
            denoise=denoise,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            tta=tta,
            temperature=temperature,
            length_penalty=length_penalty,
            auto_detect_noise=auto_detect_noise,
            num_candidates=num_candidates,
            prompt=prompt,
            fast_mode=fast_mode,
        )
        predictions.append(cap)
        used_image_paths.append(img_path)

    # Filter references to match skipped images
    ref_map = {p: r for p, r in zip(image_paths, references)}
    references_used = [ref_map[p] for p in used_image_paths]

    results = {}

    # ---- BLEU (sacrebleu) ----
    try:
        import sacrebleu

        # sacrebleu expects list of refs: [ref1_list, ref2_list, ...], each list is length N
        max_refs = max((len(r) for r in references_used), default=0)
        refs_transposed = [
            [r[j] if j < len(r) else r[-1] for r in references_used]
            for j in range(max_refs)
        ]
        bleu = sacrebleu.corpus_bleu(predictions, refs_transposed)
        results["bleu"] = float(bleu.score)
    except Exception as e:
        results["bleu_error"] = f"{type(e).__name__}: {e}"

    # ---- ROUGE (rouge-score) ----
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        r1_f, r2_f, rL_f = [], [], []

        for pred, refs in zip(predictions, references_used):
            # Best-over-references (common practice with multi-ref caption datasets)
            best1 = best2 = bestL = 0.0
            for ref in refs:
                scores = scorer.score(ref, pred)
                best1 = max(best1, scores["rouge1"].fmeasure)
                best2 = max(best2, scores["rouge2"].fmeasure)
                bestL = max(bestL, scores["rougeL"].fmeasure)
            r1_f.append(best1)
            r2_f.append(best2)
            rL_f.append(bestL)

        results["rouge1_f"] = float(np.mean(r1_f)) if r1_f else 0.0
        results["rouge2_f"] = float(np.mean(r2_f)) if r2_f else 0.0
        results["rougeL_f"] = float(np.mean(rL_f)) if rL_f else 0.0
    except Exception as e:
        results["rouge_error"] = f"{type(e).__name__}: {e}"

    # ---- METEOR (nltk) ----
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score

        # ensure tokenizers are available
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        meteor_scores = []
        for pred, refs in zip(predictions, references_used):
            # meteor_score expects tokenized strings
            pred_tok = nltk.word_tokenize(pred)
            best = 0.0
            for ref in refs:
                ref_tok = nltk.word_tokenize(ref)
                best = max(best, float(meteor_score([ref_tok], pred_tok)))
            meteor_scores.append(best)
        results["meteor"] = float(np.mean(meteor_scores)) if meteor_scores else 0.0
    except Exception as e:
        results["meteor_error"] = f"{type(e).__name__}: {e}"

    return {
        "num_images_requested": len(image_paths),
        "num_images_scored": len(predictions),
        "num_images_missing": missing,
        "split": split,
        "limit_images": limit_images,
        "seed": seed,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "num_beams": num_beams,
            "tta": tta,
            "temperature": temperature,
            "length_penalty": length_penalty,
            "auto_detect_noise": auto_detect_noise,
            "num_candidates": num_candidates,
            "prompt": prompt,
            "fast_mode": fast_mode,
        },
        "metrics": results,
    }

# removed unconditional test run so script runs only when invoked
# ...existing code...
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate caption for an image.")
    parser.add_argument("image", nargs="?", help="Path to image file")
    parser.add_argument("--denoise", choices=["none", "nlm", "bilateral", "median", "sharpen"], default="none",
                        help="Denoising method to apply before captioning (nlm requires OpenCV).")
    parser.add_argument("--beams", type=int, default=1, help="Number of beams for beam search (>=2 enables beam search, higher is slower).")
    parser.add_argument("--max-tokens", type=int, default=40, help="Max tokens to generate for the caption (lower is faster).")
    tta_group = parser.add_mutually_exclusive_group()
    tta_group.add_argument("--tta", dest="tta", action="store_true", help="Enable TTA (multiple enhanced variants).")
    tta_group.add_argument("--no-tta", dest="tta", action="store_false", help="Disable TTA (single image).")
    parser.set_defaults(tta=False)
    parser.add_argument("--no-auto-detect", action="store_true", help="Disable automatic noise detection.")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature (lower = more deterministic).")
    parser.add_argument("--length-penalty", type=float, default=1.5, help="Length penalty (>1 encourages longer captions).")
    parser.add_argument("--num-candidates", type=int, default=1, help="Number of candidate captions to generate per variant (higher is slower).")
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt prefix to guide generation.")
    parser.add_argument("--fast", action="store_true", help="Fast mode: fewer variants and prompts for quicker generation.")
    parser.set_defaults(fast=True)
    parser.add_argument("--no-speak", dest="speak", action="store_false", help="Disable speaking the generated caption aloud.")
    parser.set_defaults(speak=True)
    parser.add_argument("--eval", action="store_true", help="Run evaluation on Flickr30k pairs CSV and print caption metrics.")
    parser.add_argument("--pairs-csv", type=str, default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed", "flickr30k_pairs.csv"),
                        help="Path to flickr30k_pairs.csv (contains references + split).")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate.")
    parser.add_argument("--limit", type=int, default=500, help="Number of images to evaluate (random sample).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling eval images.")
    parser.add_argument("--eval-fast", action="store_true",
                        help="Force fast eval settings (recommended): --no-tta, --beams 1, --max-tokens 30, --num-candidates 1.")
    args = parser.parse_args()

    if args.eval:
        print("Running evaluation...", flush=True)
        if args.eval_fast:
            args.tta = False
            args.beams = 1
            args.max_tokens = 30
            args.num_candidates = 1
        report = evaluate_on_flickr30k_pairs(
            pairs_csv_path=args.pairs_csv,
            split=args.split,
            limit_images=args.limit,
            seed=args.seed,
            denoise=(None if args.denoise == "none" else args.denoise),
            max_new_tokens=args.max_tokens,
            num_beams=args.beams,
            tta=args.tta,
            temperature=args.temperature,
            length_penalty=args.length_penalty,
            auto_detect_noise=not args.no_auto_detect,
            num_candidates=max(1, args.num_candidates if args.fast else 1),
            prompt=args.prompt,
            fast_mode=True if args.fast else False,
        )
        print("\n=== Caption Evaluation Report ===")
        print(f"split: {report['split']}")
        print(f"images scored: {report['num_images_scored']} / {report['num_images_requested']} (missing files skipped: {report['num_images_missing']})")
        print("metrics:")
        for k, v in report["metrics"].items():
            print(f"  {k}: {v}")
        sys.exit(0)

    img_path = args.image
    if not img_path:
        # open file dialog
        root = tk.Tk()
        root.withdraw()
        img_path = filedialog.askopenfilename(
            title="Select image for captioning",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*")]
        )
        root.destroy()

    if not img_path:
        print("No image selected. Exiting.")
        sys.exit(1)

    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}")
        sys.exit(1)

    caption, image = generate_caption(
        img_path,
        denoise=(None if args.denoise == "none" else args.denoise),
        max_new_tokens=args.max_tokens,
        num_beams=args.beams,
        tta=args.tta,
        temperature=args.temperature,
        length_penalty=args.length_penalty,
        auto_detect_noise=not args.no_auto_detect,
        num_candidates=args.num_candidates,
        prompt=args.prompt,
        fast_mode=args.fast
    )

    print(f"\nCaption: {caption}")

    if getattr(args, "speak", False):
        speak_caption(caption)

    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Caption: {caption}", fontsize=12, wrap=True)
    plt.tight_layout()
    plt.show()