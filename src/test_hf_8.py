import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

model_id = "dishu390/visionark-model"

processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to("cpu")

try:
    model.text_decoder.cls.predictions.decoder.weight = model.text_decoder.bert.embeddings.word_embeddings.weight
except Exception as e:
    pass
model.eval()

# Test with a completely black image (which might happen if denoise fails)
image = Image.new('RGB', (224, 224), color='black')
inputs = processor(images=image, return_tensors="pt").to("cpu")

gen_kwargs = {
    "max_new_tokens": 60,
    "num_beams": 5,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
    "length_penalty": 1.5,
    "temperature": 0.9,
    "do_sample": True
}

with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Black image caption:", caption)

# Test with a completely white image
image = Image.new('RGB', (224, 224), color='white')
inputs = processor(images=image, return_tensors="pt").to("cpu")
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
print("White image caption:", processor.decode(outputs[0], skip_special_tokens=True))

# Test with random noise image
import numpy as np
noise = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
image = Image.fromarray(noise)
inputs = processor(images=image, return_tensors="pt").to("cpu")
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
print("Noise image caption:", processor.decode(outputs[0], skip_special_tokens=True))
