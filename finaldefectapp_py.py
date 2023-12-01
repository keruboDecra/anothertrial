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
def predict_metal_and_defect(image_path, metal_model, defect_model):
    # Resize the image for metal classification
    metal_img = resize_image(image_path, target_size=(224, 224))
    metal_img_array = image.img_to_array(metal_img)
    metal_img_array = np.expand_dims(metal_img_array, axis=0)
    metal_img_array /= 255.0

    # Predict metal class
    metal_prediction = metal_model.predict(metal_img_array)

    # Check if it's a metal
    is_metal = metal_prediction[0][0] > 0.5  # Adjust the threshold if needed

    if is_metal:
        # Resize the image for defect prediction
        defect_img = resize_image(image_path, target_size=(150, 150))
        defect_img_array = image.img_to_array(defect_img)
        defect_img_array = np.expand_dims(defect_img_array, axis=0)
        defect_img_array /= 255.0

        # Predict defect class with probability scores
        defect_prediction = defect_model.predict(defect_img_array)
        defect_class = np.argmax(defect_prediction)
        defect_probabilities = defect_prediction[0]

        return "Metal", defect_class, defect_probabilities
    else:
        return "Non-Metal", None, None

# Streamlit app
st.title("Defects Assessment App")

# Load the models
metal_classification_model = load_model('classifymaterial.h5')
defect_prediction_model = load_model('mobilenet_model (1).h5')

# Defect class names mapping
defect_class_names = {
    0: "Pitted",
    1: "Inclusion",
    2: "Crazing",
    3: "Patches",
    4: "Scratches",
    5: "Rolled"
}

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Get predictions
    metal_label, defect_label, defect_probabilities = predict_metal_and_defect(uploaded_file, metal_classification_model, defect_prediction_model)

    # Display results
    st.write(f"Metal Classification: {metal_label}")
    if metal_label == "Metal" and defect_label is not None:
        defect_class_name = defect_class_names.get(defect_label, "Unknown")
        st.write(f"Defect Classification: {defect_class_name}")
        st.write("Defect Probabilities:")
        for i, prob in enumerate(defect_probabilities):
            st.write(f"{defect_class_names[i]}: {prob:.4f}")
