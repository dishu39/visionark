import torch
from transformers import BlipForConditionalGeneration

model_id = "dishu390/visionark-model"
model = BlipForConditionalGeneration.from_pretrained(model_id, tie_word_embeddings=True)

print("Are weights tied?", torch.equal(model.text_decoder.cls.predictions.decoder.weight, model.text_decoder.bert.embeddings.word_embeddings.weight))
