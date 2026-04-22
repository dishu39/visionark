import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

model_id = "dishu390/visionark-model"

try:
    processor = BlipProcessor.from_pretrained(model_id)
    print("Processor loaded. Vocab size:", processor.tokenizer.vocab_size)
except Exception as e:
    print("Error loading processor:", e)

try:
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    print("Model loaded.")
except Exception as e:
    print("Error loading model:", e)
