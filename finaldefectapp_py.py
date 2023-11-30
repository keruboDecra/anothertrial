# Importing necessary libraries
import streamlit as st
import os
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

# Function to load the trained MobileNet model
def load_mobilenet_model():
    model_path = 'mobilenet_model (1).h5'

    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Function to make predictions
def predict_defect(image_path, model):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    return prediction

# Function to assess the highest probability predicted and print out the class of the image
def assess_defect(prediction, classes):
    max_prob_index = np.argmax(prediction)
    max_prob_class = classes[max_prob_index]
    return max_prob_class

# Streamlit App
def main():
    st.title("Defects Assessment App")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        # Create a temporary directory if it doesn't exist
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)

        # Create a path for the temporary image
        temp_path = os.path.join(temp_dir, 'temp_image.jpg')
        uploaded_file.seek(0)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Load the model
        model = load_mobilenet_model()

        if model is not None:
            # Make predictions
            prediction = predict_defect(temp_path, model)

            # Display the results
            st.subheader("Prediction Results:")
            for i, class_name in enumerate(classes):
                st.write(f"{class_name}: {prediction[0][i]}")

            # Assess the highest probability predicted and print out the class
            max_prob_class = assess_defect(prediction[0], classes)
            st.success(f"This metal surface has a defect of: {max_prob_class}")

            # Set a threshold for alerting
            threshold = 0.95 # Set your chosen threshold
            max_prob = max(prediction[0])
            if max_prob < threshold:
                st.warning("No relevant defect found. Please check the image again.")

# Define your classes
classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Run the app
if __name__ == '__main__':
    main()
