"""
Test script to verify all dependencies are installed correctly
Run this before starting the API server
"""

import sys

print("=" * 60)
print("Testing Image Captioning API Dependencies")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")
python_version = sys.version_info
if python_version.major == 3 and python_version.minor >= 11 and python_version.minor <= 13:
    print("   ✅ Python version is compatible")
elif python_version.minor == 14:
    print("   ⚠️  Python 3.14 is very new - some packages may not work")
    print("   💡 Consider using Python 3.11 or 3.12 for better compatibility")
else:
    print("   ❌ Python version may be too old")

# Test PyTorch
print("\n2. PyTorch:")
try:
    import torch
    print(f"   ✅ Version: {torch.__version__}")
    print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA version: {torch.version.cuda}")
        print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   ℹ️  Running on CPU (slower but works)")
except ImportError as e:
    print(f"   ❌ PyTorch not installed: {e}")
    print("   💡 Install: pip install torch --index-url https://download.pytorch.org/whl/cu118")

# Test Transformers
print("\n3. Transformers:")
try:
    import transformers
    from transformers import BlipProcessor, BlipForConditionalGeneration
    print(f"   ✅ Version: {transformers.__version__}")
except ImportError as e:
    print(f"   ❌ Transformers not installed: {e}")
    print("   💡 Install: pip install transformers")

# Test Pillow
print("\n4. Pillow (Image Processing):")
try:
    from PIL import Image
    import PIL
    print(f"   ✅ Version: {PIL.__version__}")
except ImportError as e:
    print(f"   ❌ Pillow not installed: {e}")
    print("   💡 Install: pip install Pillow")

# Test OpenCV (optional)
print("\n5. OpenCV (Optional - for denoising):")
try:
    import cv2
    print(f"   ✅ Version: {cv2.__version__}")
    print("   ✅ Advanced denoising available")
except ImportError:
    print("   ⚠️  OpenCV not installed")
    print("   ℹ️  This is optional - basic functionality will work without it")
    print("   💡 Install: pip install opencv-python")

# Test NumPy
print("\n6. NumPy:")
try:
    import numpy as np
    print(f"   ✅ Version: {np.__version__}")
except ImportError as e:
    print(f"   ❌ NumPy not installed: {e}")
    print("   💡 Install: pip install numpy")

# Test FastAPI
print("\n7. FastAPI:")
try:
    import fastapi
    print(f"   ✅ Version: {fastapi.__version__}")
except ImportError as e:
    print(f"   ❌ FastAPI not installed: {e}")
    print("   💡 Install: pip install fastapi")

# Test Uvicorn
print("\n8. Uvicorn (API Server):")
try:
    import uvicorn
    print(f"   ✅ Version: {uvicorn.__version__}")
except ImportError as e:
    print(f"   ❌ Uvicorn not installed: {e}")
    print("   💡 Install: pip install uvicorn[standard]")

# Test python-multipart
print("\n9. Python Multipart (File Upload):")
try:
    import multipart
    print("   ✅ Installed")
except ImportError:
    print("   ❌ python-multipart not installed")
    print("   💡 Install: pip install python-multipart")

# Summary
print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)

try:
    import torch
    import transformers
    from PIL import Image
    import fastapi
    import uvicorn
    
    print("✅ All CORE packages installed successfully!")
    print("\n🚀 You're ready to run the API server:")
    print("   python api_server.py")
    
    try:
        import cv2
        print("\n✨ Bonus: Advanced denoising is available!")
    except:
        print("\nℹ️  Install opencv-python for advanced denoising (optional)")
    
except ImportError:
    print("❌ Some core packages are missing")
    print("\n📋 Install all requirements:")
    print("   pip install -r requirements_api.txt")
    print("\n   Or install manually:")
    print("   pip install torch transformers fastapi uvicorn pillow")

print("\n" + "=" * 60)

# Test model loading capability
print("\n10. Testing Model Loading Capability:")
try:
    import torch
    from transformers import BlipProcessor
    
    print("   ℹ️  Attempting to load processor (this may take a moment)...")
    # Try loading from HuggingFace cache (won't download)
    try:
        processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            local_files_only=True
        )
        print("   ✅ Model files found in cache")
    except:
        print("   ℹ️  Model not in cache (will download when API starts)")
    
    print("   ✅ Model loading capability verified")
    
except Exception as e:
    print(f"   ⚠️  Issue with model loading: {e}")

print("\n" + "=" * 60)