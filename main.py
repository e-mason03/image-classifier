import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

def load_model():
    return MobileNetV2(weights="imagenet")

def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

def classify_image(model, image):
    try:
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        # Return the top 3 predictions
        return decode_predictions(predictions, top=3)[0]
    
    except Exception as e:
        st.error(f"Error classifying image: {str(e)}")
        return None
    
def main():
    # Set up UI configuration
    st.set_page_config(
        page_title="AI Image Classifier", page_icon="🔎", layout="centered"
    )
    st.title("AI Image Classifier")
    st.write("Upload an image and let AI tell you what's in it!")

    # Cache the model to avoid reloading it upon app refresh
    @st.cache_resource
    def load_cached_model():
        return load_model()
    
    model = load_cached_model()
    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png"])

    # On upload, render the button to trigger classification
    if uploaded_file is not None:
        image = st.image(
            uploaded_file, caption="Uploaded Image", use_container_width=True
        )
        btn = st.button("Classify Image")
        # When the button is pressed, classify the image
        if btn:
            with st.spinner("Classifying Image..."):
                image = Image.open(uploaded_file)
                predictions = classify_image(model, image)
                # Print the top 3 predictions
                if predictions:
                    st.subheader("Predictions:")
                    for _, label, score in predictions:
                        st.write(f"**{label}**: {score:.2%}")

if __name__ == "__main__":
    main()