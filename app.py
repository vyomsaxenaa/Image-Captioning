# -*- coding: utf-8 -*-
!pip install transformers

from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

model= VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
tokenizer=AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4

gen_kwargs = {"max_length":max_length, "num_beams": num_beams}

def predict_step(images):

  pixel_values=feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values=pixel_values.to(device)

  output_ids=model.generate(pixel_values, **gen_kwargs)

  preds=tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds=[pred.strip() for pred in preds]

  return preds

!pip install gradio

import gradio as gr

inputs=[
    gr.inputs.Image(type="pil", label="ORiginal Image")
]

outputs=[
    gr.outputs.Textbox(label="Caption")
]

title="Image Captioning"
description="AI based Caption generator"
article = "<a href = 'https://huggingface.co/nlpconnect/vit-gpt2-image-captioning'>Model Repo hugging face model hub</a>"
examples = [
    ["Image1.png"]
]

gr.Interface(
    predict_step,
    inputs,
    outputs,
    title=title,
    description=description,
    article=article,
    examples=examples,
    theme="huggingFace"

).launch(debug=True, enable_queue=True)
