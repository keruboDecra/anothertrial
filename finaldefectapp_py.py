import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


# Load models
metal_classification_model = load_model('classifymaterial.h5')
defect_prediction_model = load_model('mobilenet_model (1).h5')

# Class labels for metal classification
metal_classes = ['plastic', 'paper', 'organic', 'metal', 'light blubs', 'glass', 'e-waste', 'clothes', 'batteries']

# Class labels for defect prediction
defect_classes = ['Pitted', 'Inclusion', 'Crazing', 'Patches', 'Scratches', 'Rolled']

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


# Streamlit app
def main():
    st.title("Metal Defect Prediction App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        # Preprocess the image
        img_array = preprocess_image(uploaded_file)

        # Metal classification
        metal_probabilities = metal_classification_model.predict(img_array)[0]
        predicted_metal_class = metal_classes[np.argmax(metal_probabilities)]
        st.write(f"Predicted Material Class: {predicted_metal_class}")

        # Check if the material is metal
        if predicted_metal_class.lower() == 'metal':
            # Defect prediction
            defect_probabilities = defect_prediction_model.predict(img_array)[0]
            predicted_defect_class = defect_classes[np.argmax(defect_probabilities)]

            st.write("Defect Prediction:")
            for i in range(len(defect_classes)):
                st.write(f"{defect_classes[i]}: {defect_probabilities[i]:.2%}")

            st.write(f"Predicted Defect Class: {predicted_defect_class}")

if __name__ == "__main__":
    main()
