from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import os
from pathlib import Path
import base64
from typing import Optional
import numpy as np
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Try importing cv2 for denoising
try:
    import cv2
    _HAS_CV2 = True
except:
    _HAS_CV2 = False

# ===== CONFIG =====
# Configured for Render deployment
MODEL_NAME = "dishu390/visionark-model"
device = "cpu"

# Global references for the model
processor = None
model = None

# ===== FASTAPI APP =====
app = FastAPI(
    title="VISIONARK API",
    description="Generate captions for images using fine-tuned BLIP model",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files will be mounted at the end of the file to not shadow API routes

# ===== SAFE MODEL LOADING ON STARTUP =====
@app.on_event("startup")
def load_model():
    global processor, model
    print("Loading model from HuggingFace...")
    print(f"Using device: {device}")
    
    try:
        processor = BlipProcessor.from_pretrained(MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
        
        # FIX: Ensure decoder weights are correctly tied to word embeddings
        # If the HF repo is missing the tie_weights config, it generates gibberish.
        try:
            model.text_decoder.cls.predictions.decoder.weight = model.text_decoder.bert.embeddings.word_embeddings.weight
        except Exception as e:
            print(f"Warning: Could not tie weights: {e}")
            
        model.eval()  # Optimize for low memory and inference
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        processor = None
        model = None

# ===== HELPER FUNCTIONS =====

def denoise_image(pil_img, method="nlm", strength="medium"):
    """Denoise image with various methods"""
    if method is None or method.lower() == "none":
        return pil_img
    
    if _HAS_CV2 and method.lower() in ("nlm", "bilateral"):
        arr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        if method.lower() == "nlm":
            if strength == "light":
                h, hColor = 10, 10
            elif strength == "strong":
                h, hColor = 20, 20
            else:  # medium
                h, hColor = 15, 15
            den = cv2.fastNlMeansDenoisingColored(arr, None, h, hColor, 7, 21)
        else:  # bilateral
            if strength == "light":
                d, sigmaColor, sigmaSpace = 5, 50, 50
            elif strength == "strong":
                d, sigmaColor, sigmaSpace = 11, 100, 100
            else:
                d, sigmaColor, sigmaSpace = 9, 75, 75
            den = cv2.bilateralFilter(arr, d, sigmaColor, sigmaSpace)
        
        den = cv2.cvtColor(den, cv2.COLOR_BGR2RGB)
        return Image.fromarray(den)
    
    return pil_img

def detect_noise_level(pil_img):
    """Detect noise level in image"""
    if not _HAS_CV2:
        return "medium"
    
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    patches = []
    for _ in range(20):
        y = np.random.randint(0, max(1, h - 20))
        x = np.random.randint(0, max(1, w - 20))
        patch = arr[y:y+20, x:x+20]
        patches.append(np.var(patch))
    
    avg_variance = np.mean(patches)
    
    if avg_variance > 2500:
        return "strong"
    elif avg_variance > 1200:
        return "medium"
    else:
        return "light"

@torch.no_grad()
def generate_caption_fast(
    pil_img: Image.Image,
    denoise_method: Optional[str] = None,
    max_tokens: int = 60,
    num_beams: int = 5,
    temperature: float = 0.9,
    auto_denoise: bool = True
):
    """Fast caption generation optimized for low memory"""
    global processor, model
    
    if auto_denoise and denoise_method is None:
        noise_level = detect_noise_level(pil_img)
        if noise_level in ("strong", "aggressive"):
            pil_img = denoise_image(pil_img, method="nlm", strength=noise_level)
    elif denoise_method and denoise_method != "none":
        pil_img = denoise_image(pil_img, method=denoise_method, strength="medium")
    
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "num_beams": num_beams,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "length_penalty": 1.5,
        "temperature": temperature,
        "do_sample": True
    }
    
    outputs = model.generate(**inputs, **gen_kwargs)
    caption = processor.decode(outputs[0], skip_special_tokens=True)
    
    return caption

# ===== API ENDPOINTS =====

# Home route handled by StaticFiles at the root

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }

@app.post("/caption")
async def generate_caption_endpoint(
    file: UploadFile = File(...),
    denoise: Optional[str] = Form(None),
    max_tokens: int = Form(60),
    num_beams: int = Form(5),
    temperature: float = Form(0.9),
    auto_denoise: bool = Form(True)
):
    """Generate caption for uploaded image with time tracking and safe error handling"""
    
    if model is None or processor is None:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Model not loaded properly on server startup"}
        )
    
    start_time = time.time()
    
    try:
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Invalid image file provided"}
            )
        
        caption = generate_caption_fast(
            pil_img=image,
            denoise_method=denoise,
            max_tokens=max_tokens,
            num_beams=num_beams,
            temperature=temperature,
            auto_denoise=auto_denoise
        )
        
        time_taken = round(time.time() - start_time, 2)
        
        return JSONResponse({
            "success": True,
            "caption": caption,
            "time_taken": time_taken
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": f"Error processing request: {str(e)}"}
        )

@app.post("/caption-base64")
async def generate_caption_base64(
    image_base64: str = Form(...),
    denoise: Optional[str] = Form(None),
    max_tokens: int = Form(60),
    num_beams: int = Form(5),
    temperature: float = Form(0.9),
    auto_denoise: bool = Form(True)
):
    if model is None or processor is None:
        return JSONResponse(status_code=503, content={"success": False, "error": "Model not loaded"})
    
    start_time = time.time()
    
    try:
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        try:
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return JSONResponse(status_code=400, content={"success": False, "error": "Invalid base64 payload"})
        
        caption = generate_caption_fast(
            pil_img=image,
            denoise_method=denoise,
            max_tokens=max_tokens,
            num_beams=num_beams,
            temperature=temperature,
            auto_denoise=auto_denoise
        )
        
        time_taken = round(time.time() - start_time, 2)
        
        return JSONResponse({
            "success": True,
            "caption": caption,
            "time_taken": time_taken
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": f"Error processing image: {str(e)}"})

@app.post("/batch-caption")
async def batch_caption(
    files: list[UploadFile] = File(...),
    denoise: Optional[str] = Form(None),
    max_tokens: int = Form(60),
    num_beams: int = Form(3),
    temperature: float = Form(0.9),
    auto_denoise: bool = Form(True)
):
    if model is None or processor is None:
        return JSONResponse(status_code=503, content={"success": False, "error": "Model not loaded"})
    
    start_time = time.time()
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            caption = generate_caption_fast(
                pil_img=image,
                denoise_method=denoise,
                max_tokens=max_tokens,
                num_beams=num_beams,
                temperature=temperature,
                auto_denoise=auto_denoise
            )
            
            results.append({
                "filename": file.filename,
                "caption": caption,
                "success": True
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "caption": None,
                "success": False,
                "error": str(e)
            })
            
    time_taken = round(time.time() - start_time, 2)
    
    return JSONResponse({
        "success": True,
        "results": results,
        "total": len(files),
        "successful": sum(1 for r in results if r["success"]),
        "time_taken": time_taken
    })

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("static/index.html")

# ===== ENTRY POINT FOR DEPLOYMENT =====
if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)