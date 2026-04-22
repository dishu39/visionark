import torch
from transformers import BlipForConditionalGeneration

model_id = "dishu390/visionark-model"
base_id = "Salesforce/blip-image-captioning-base"

print("Loading models...")
model = BlipForConditionalGeneration.from_pretrained(model_id).to("cpu")
base = BlipForConditionalGeneration.from_pretrained(base_id).to("cpu")

diff_count = 0
total_count = 0
for (n1, p1), (n2, p2) in zip(model.named_parameters(), base.named_parameters()):
    if p1.shape != p2.shape:
        print(f"Shape mismatch at {n1}: {p1.shape} vs {p2.shape}")
        continue
    
    if not torch.allclose(p1, p2, atol=1e-4):
        #print(f"Weights differ: {n1}")
        diff_count += 1
    total_count += 1

print(f"{diff_count} out of {total_count} layers differ.")
