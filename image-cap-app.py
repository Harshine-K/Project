import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration
import gradio as gr
import numpy as np

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def image_cap(input_image: np.ndarray):
    raw_image = Image.fromarray(input_image).convert('RGB')
    inputs = processor(raw_image, return_tensors='pt')
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

iface = gr.Interface(fn=image_cap,
                     inputs=gr.Image(),
                     outputs="text",
                     title="Image Captioning Application",
                     description="This is a simple web app to generte captions for images")
iface.launch(server_name="127.0.0.1", server_port=7860)