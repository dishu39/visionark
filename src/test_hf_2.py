import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

model_id = "dishu390/visionark-model"

model = BlipForConditionalGeneration.from_pretrained(model_id)

print("Before tying:", torch.equal(model.text_decoder.cls.predictions.decoder.weight, model.text_decoder.bert.embeddings.word_embeddings.weight))

# Manually tie weights
model.text_decoder.cls.predictions.decoder.weight = model.text_decoder.bert.embeddings.word_embeddings.weight

print("After tying:", torch.equal(model.text_decoder.cls.predictions.decoder.weight, model.text_decoder.bert.embeddings.word_embeddings.weight))

processor = BlipProcessor.from_pretrained(model_id)

# Test generation
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=20)
print("Caption:", processor.decode(out[0], skip_special_tokens=True))
