import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load the metal classification model
metal_classification_model = load_model('classifyWaste.h5')

# Load the MobileNet defect prediction model
defect_prediction_model = load_model('mobilenet_model (1).h5')

# Function to predict metal and defect
def predict_metal_and_defect(image_path):
    # Load and preprocess the image for metal classification
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict metal class
    metal_prediction = metal_classification_model.predict(img_array)

    # Check if it's a metal
    is_metal = metal_prediction[0][0] > 0.5  # Adjust the threshold if needed

    if is_metal:
        # Load and preprocess the image for defect prediction
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Predict defect class
        defect_prediction = defect_prediction_model.predict(img_array)
        defect_class = np.argmax(defect_prediction)

        return "Metal", defect_class
    else:
        return "Non-Metal", None

# Streamlit app
st.title("Metal and Defect Prediction App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Make predictions
    metal_label, defect_label = predict_metal_and_defect(uploaded_file)

    # Display predictions
    st.write(f"Predicted Metal or Non-Metal: {metal_label}")
    
    if defect_label is not None:
        st.write(f"Predicted Defect Class: {defect_label}")
