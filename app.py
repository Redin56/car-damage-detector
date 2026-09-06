import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

label_map = {0: 'Damaged', 1: 'Clean'} #prefered clean here because its faster to grasp

def predict(image):
  image = image.convert('RGB')
  image = transforms(image).unsqueeze(0) # adds a batch dimension because thats what the model expects

  with torch.no_grad():
    output = model(image)
    probs = torch.softmax(output, dim=1)
    conf, pred = torch.max(probs, dim=1)
  
  label = label_map[pred.item()] #turns numeric class to readable label
  confidence = conf.item() * 100
  return label, confidence

st.title('Car Damage Classifier')
st.write('Upload an image of a car and the model will check whether it is damaged or clean.')

uploaded_file = st.file_uploader('Upload an image', type=['jpg','jpeg', 'png', 'webp','bmp'])

if uploaded_file is not None:
  image =Image.open(uploaded_file)
  st.image(image, caption = 'Uploaded Image', use_container_width = True)

  label, confidence = predict(image)
  st.write(f'Prediction: {label} ({confidence:.2f}% Confidence)')
