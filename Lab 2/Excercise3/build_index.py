import datasets
from datasets import load_dataset

import torch
import numpy as np

from transformers import CLIPProcessor, CLIPModel


datasets.config.IN_MEMORY_MAX_SIZE = 0  # disabilita multiprocessing

#importo dataset
dataset = load_dataset("jxie/flickr8k", split="test", num_proc=1)  # ~1000 immagini per prototipo
# ogni sample ha: dataset[i]["image"] (PIL.Image) e dataset[i]["caption"] (list of str)

images = [sample["image"] for sample in dataset]




#importo modello CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model.eval()


import torch.nn.functional as F


#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
#creo indice
def index_images(images, batch_size=64):
    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        inputs = processor(images=batch, return_tensors="pt", padding=True)
        with torch.no_grad():
            # bypass get_image_features, estrai direttamente il tensore
            vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
            pooled = vision_out.pooler_output                  # (batch, 768) - tensore puro
            projected = model.visual_projection(pooled)        # (batch, 512) - tensore puro
            embeddings = F.normalize(projected, dim=-1)        # niente .norm(), usa F
        all_embeddings.append(embeddings.cpu().numpy())
        print(f"{min(i+batch_size, len(images))}/{len(images)}", end="\r")
    return np.vstack(all_embeddings)

image_index = index_images(images)
np.save("image_index.npy", image_index)        #salvo per non ricalcolare ogni volta




