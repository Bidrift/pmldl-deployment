import streamlit as st
import requests
from PIL import Image
import io

st.title("Image Prediction App")
st.write("Upload an image to get a prediction.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if st.button("Predict"):
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='JPEG')
        image_bytes.seek(0)
        files = {'file': image_bytes}
        response = requests.post("http://api:8000/predict", files=files)
        
        if response.status_code == 200:
            prediction = response.json()
            st.write(f"Prediction: {prediction['prediction']}")
        else:
            st.write("Error: Failed to get a prediction from the model API.")
