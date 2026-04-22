import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

model_id = "dishu390/visionark-model"

model = BlipForConditionalGeneration.from_pretrained(model_id)

print("Before tie_weights():", torch.equal(model.text_decoder.cls.predictions.decoder.weight, model.text_decoder.bert.embeddings.word_embeddings.weight))

model.tie_weights()

print("After tie_weights():", torch.equal(model.text_decoder.cls.predictions.decoder.weight, model.text_decoder.bert.embeddings.word_embeddings.weight))
