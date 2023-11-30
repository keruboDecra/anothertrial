# Importing necessary libraries
import streamlit as st
import os
from pathlib import Path
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
from skimage.metrics import structural_similarity as ssim
from skimage import io, color

# Function to load the trained MobileNet model
def load_mobilenet_model(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None

# Function to calculate Structural Similarity Index (SSI)
def calculate_ssim(reference_image, image_path):
    # Read the reference image
    reference_img = io.imread(reference_image)
    
    # Read the uploaded image
    uploaded_img = io.imread(image_path)
    
    # Convert images to grayscale
    reference_gray = color.rgb2gray(reference_img)
    uploaded_gray = color.rgb2gray(uploaded_img)
    
    # Calculate SSIM
    index, _ = ssim(reference_gray, uploaded_gray, full=True)
    
    return index

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
        # Create a temporary directory if it doesn't exist
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)

        # Create a path for the temporary image
        temp_path = os.path.join(temp_dir, 'temp_image.jpg')
        uploaded_file.seek(0)
        with open(temp_path, 'wb') as f:
            f.write(uploaded_file.read())

        # Reference image for SSI
        reference_image = 'rusty_metal.jpg'

        # Calculate SSIM
        ssim_index = calculate_ssim(reference_image, temp_path)

        # Set a threshold for SSIM
        ssim_threshold = 0.7  # Adjust the threshold as needed

        if ssim_index > ssim_threshold:
            # Load the model for defect assessment
            model_path = 'mobilenet_model (1).h5'
            defect_model = load_mobilenet_model(model_path)

            if defect_model is not None:
                # Make predictions for defects
                prediction = predict_defect(temp_path, defect_model)

                # Display the results
                st.subheader("Prediction Results:")
                for i, class_name in enumerate(classes):
                    st.write(f"{class_name}: {prediction[i]}")

                # Assess the highest probability predicted and print out the class
                max_prob_class = assess_defect(prediction, classes)
                st.success(f"This metal surface has a defect of: {max_prob_class}")
            else:
                st.error("Defect assessment model not loaded.")
        else:
            st.warning("The uploaded image does not appear to contain a metallic surface.")

# Define your classes
classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']

# Run the app
if __name__ == '__main__':
    main()
