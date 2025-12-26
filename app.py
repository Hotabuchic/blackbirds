import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import json

from model import BirdClassifier


MODEL_PATH = "models/model.pt"
CLASS_MAPPING_PATH = "class_mapping.json"
IMAGE_SIZE = 224


@st.cache_resource
def load_model():
    with open(CLASS_MAPPING_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)

    idx_to_class = {k: v for k, v in class_to_idx.items()}

    model = BirdClassifier(num_classes=len(idx_to_class))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model, idx_to_class


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


st.title("🕊️ Bird Species Classification")
st.write("Загрузите изображение птицы для определения вида")

model, idx_to_class = load_model()

uploaded_file = st.file_uploader(
    "Загрузите изображение",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    predicted_class = idx_to_class[str(pred_idx.item())]
    confidence = conf.item()

    st.subheader("Результат:")
    st.write(f"**Вид:** {predicted_class}")
    st.write(f"**Уверенность:** {confidence:.2%}")
