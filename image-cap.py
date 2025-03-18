import requests
from PIL import Image
from transformers import AutoProcessor, BlipForConditionalGeneration

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

img_path = "image.png"
image = Image.open(img_path).convert('RGB')

text = "The image of"
inputs = processor(images=image, text=text, return_tensors='pt')
#print(inputs)

output = model.generate(**inputs)
#print(output)

caption=processor.decode(output[0], skip_special_tokens=True)
print(caption)