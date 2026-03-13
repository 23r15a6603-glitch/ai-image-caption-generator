import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
import torch

# Page settings
st.set_page_config(
    page_title="AI Image Caption Generator",
    page_icon="🖼",
    layout="wide"
)

# Device selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Sidebar
st.sidebar.title("⚙ Settings")

language = st.sidebar.selectbox(
    "Select Caption Language",
    ["English", "Hindi", "Telugu"]
)

st.sidebar.markdown("---")
st.sidebar.info(
"""
**Model:** BLIP Image Captioning  
**Framework:** HuggingFace Transformers  

Features:
- Image Caption Generation
- Multi-language Translation
- AI Vision Model
"""
)

# Load model with caching
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    model.eval()
    return processor, model

processor, model = load_model()

# Title
st.title("🖼 AI Image Caption Generator")
st.write("Upload an image and the AI will generate a caption for it.")

st.markdown("---")

# Upload section
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# Caption generation function
def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


# If image uploaded
if uploaded_file is not None:

    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:

        st.subheader("Generate Caption")

        if st.button("Generate Caption 🚀"):

            with st.spinner("AI is generating caption..."):

                caption = generate_caption(image)

                # Translation
                if language == "Hindi":
                    caption = GoogleTranslator(source='auto', target='hi').translate(caption)

                elif language == "Telugu":
                    caption = GoogleTranslator(source='auto', target='te').translate(caption)

                st.success("Caption Generated Successfully")

                st.markdown("### 📝 Caption")
                st.info(caption)

else:
    st.warning("Please upload an image to generate a caption.")
