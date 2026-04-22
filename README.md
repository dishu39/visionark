
# 🧠 Noise-Reduced Image Captioning using Flickr30k + BLIP

This project builds a **clean, high-quality vision–language dataset** from Flickr30k and fine-tunes the **BLIP (Salesforce)** model to generate accurate captions.  
The focus is on **noise reduction** in captions before model training — resulting in a higher-quality dataset and improved caption performance.

---

## 🚀 Overview

Raw web datasets often contain:

- Noisy captions (URLs, numbers, junk)
- Multiple languages
- Mismatched captions and images
- Irrelevant descriptions

This project implements a **three-stage filtering pipeline**:

| Stage | Technique | Goal |
|-------|-----------|------|
| 1️⃣ Rule-based filtering | Regex + heuristics | Remove URLs, gibberish, spam |
| 2️⃣ Language detection | `langdetect` | Keep only English captions |
| 3️⃣ Semantic filtering | CLIP similarity scoring | Remove image–caption mismatches |

The final cleaned dataset is then used to fine-tune a **BLIP image captioning model**.

---

## 📂 Project Structure

```
vision_noise_project/
│
├── data/
│   ├── raw/
│   │   └── images/flickr30k/         
│   └── processed/
│       ├── flickr30k_pairs.csv       
│       ├── flickr30k_rule_clean.csv  
│       ├── flickr30k_lang_clean.csv  
│       └── flickr30k_clip_clean.csv  
│
├── model_final/                      
│
├── src/
│   ├── 01_prepare_dataset.py
│   ├── 02_rule_clean.py
│   ├── 03_lang_filter.py
│   ├── 04_clip_filter.py
│   ├── 05_train_model.py             
│   └── 06_inference.py               
│
├── requirements.txt
└── README.md
```

---

## 🧪 Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate     
venv\Scripts\activate        
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📥 Dataset

The project uses **Flickr30k**, cleaned using the filtering scripts.

The final supervised dataset is stored as:

```
data/processed/flickr30k_clip_clean.csv
```

---

## 🛠 Model Training

Run:

```bash
python src/05_train_model.py
```

---

## 📌 Inference

```bash
python src/06_inference.py
```

---

## 📊 Results

| Metric | Before Cleaning | After Cleaning |
|--------|----------------|----------------|
| Retrieval Recall@1 | lower | improved |
| Caption quality | noisy | meaningful |

---

## 👨‍💻 Author

Engineering Student | Machine Learning | Vision-Language Models

---

