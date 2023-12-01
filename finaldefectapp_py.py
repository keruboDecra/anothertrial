import os
import numpy as np
from PIL import Image
from keras.preprocessing import image
from keras.models import load_model
import streamlit as st

# Function to resize an image
def resize_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path)
    img = img.resize(target_size)
    return img

# Function to predict metal and defect
def predict_metal_and_defect(image_path):
    # Resize the image for metal classification
    metal_img = resize_image(image_path, target_size=(224, 224))
    metal_img_array = image.img_to_array(metal_img)
    metal_img_array = np.expand_dims(metal_img_array, axis=0)
    metal_img_array /= 255.0

    # Predict metal class
    metal_prediction = metal_classification_model.predict(metal_img_array)

    # Check if it's a metal
    is_metal = metal_prediction[0][0] > 0.1  # Adjust the threshold if needed

    if is_metal:
        # Resize the image for defect prediction
        defect_img = resize_image(image_path, target_size=(150, 150))
        defect_img_array = image.img_to_array(defect_img)
        defect_img_array = np.expand_dims(defect_img_array, axis=0)
        defect_img_array /= 255.0

        # Predict defect class
        defect_prediction = defect_prediction_model.predict(defect_img_array)
        defect_class = np.argmax(defect_prediction)

        return "Metal", defect_class
    else:
        return "Non-Metal", None

# Load the models
metal_classification_model = load_model('classifyWaste.h5')
defect_prediction_model = load_model('mobilenet_model (1).h5')

# Streamlit app
st.title("Defects Assessment App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Get predictions
    metal_label, defect_label = predict_metal_and_defect(uploaded_file)

    # Display results
    st.write(f"Metal Classification: {metal_label}")
    if metal_label == "Metal" and defect_label is not None:
        st.write(f"Defect Classification: Class {defect_label}")
