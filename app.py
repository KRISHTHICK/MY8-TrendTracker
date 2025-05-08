import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="TrendTracker", layout="wide")
st.title("📈 TrendTracker - Real-Time Fashion Trend Detector")

uploaded_image = st.file_uploader("📸 Upload Fashion Image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🔍 Detecting fashion elements..."):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        inputs = processor(images=image, return_tensors="pt")
        output = model.generate(**inputs)
        description = processor.decode(output[0], skip_special_tokens=True)

    st.subheader("🧾 Detected Fashion Description:")
    st.write(description)

    st.subheader("📊 Trend Analysis & Recommendations")
    prompt = PromptTemplate.from_template(
        "Based on the following fashion description: '{desc}', analyze if the items are currently trending, rising, or outdated. Suggest one outfit improvement and generate a catchy Instagram caption with 5 trending hashtags."
    )
    final_prompt = prompt.format(desc=description)

    llm = OpenAI(temperature=0.7)
    result = llm(final_prompt)
    st.text_area("📍 AI Trend Analysis:", result, height=250)
