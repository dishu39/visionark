import torch
from transformers import BlipForConditionalGeneration

model_id = "dishu390/visionark-model"
model = BlipForConditionalGeneration.from_pretrained(model_id).to("cpu")

bias = model.text_decoder.cls.predictions.bias
print("Bias norm:", bias.norm().item())
print("Bias max:", bias.max().item())
print("Bias min:", bias.min().item())
