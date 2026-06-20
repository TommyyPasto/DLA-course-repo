from datasets import load_dataset

import torch
import numpy as np

from transformers import CLIPProcessor, CLIPModel


#importo dataset
dataset = load_dataset("jxie/flickr8k", split="test")  
images = [sample["image"] for sample in dataset]


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
model.eval()


import torch.nn.functional as F

def retrieve(query: str, image_index: np.ndarray, top_k: int = 10):
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_out = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )
        pooled = text_out.pooler_output                  # tensore puro
        text_emb = model.text_projection(pooled)         # tensore puro
        text_emb = F.normalize(text_emb, dim=-1).cpu().numpy()

    scores = (image_index @ text_emb.T).squeeze()
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]



import gradio as gr

image_index = np.load("image_index.npy")

def search(query):
    indeces, scores = retrieve(query, image_index, top_k=10)
    return[(images[i], f"score: {scores[j]:.3f}") for j, i in enumerate(indeces)]

with gr.Blocks() as app:
    gr.Markdown("## CLIP Image Search Engine")
    with gr.Row():
        query_input = gr.Textbox(label="Search query", placeholder="e.g. 'a dog playing in the park'")
        search_button = gr.Button("Search")
    results_gallery = gr.Gallery(label="Results", columns=5, height="auto")
    search_button.click(search, inputs=query_input, outputs=results_gallery)
    
app.launch()