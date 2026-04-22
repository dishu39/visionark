import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import requests
import io

model_id = "dishu390/visionark-model"

processor = BlipProcessor.from_pretrained(model_id)
model = BlipForConditionalGeneration.from_pretrained(model_id).to("cpu")

try:
    model.text_decoder.cls.predictions.decoder.weight = model.text_decoder.bert.embeddings.word_embeddings.weight
except Exception as e:
    print(e)
    
model.eval()

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

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

print("Generated with API server kwargs:", caption)

gen_kwargs_2 = {
    "max_new_tokens": 60,
    "num_beams": 5,
    "length_penalty": 1.2, # Encourages detail
    "repetition_penalty": 1.5,
    "no_repeat_ngram_size": 3
}

with torch.no_grad():
    outputs2 = model.generate(**inputs, **gen_kwargs_2)
caption2 = processor.decode(outputs2[0], skip_special_tokens=True)

print("Generated with 07_inference kwargs:", caption2)
