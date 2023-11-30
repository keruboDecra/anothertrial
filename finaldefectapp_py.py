# Importing necessary libraries
import streamlit as st
import os
from pathlib import Path
from keras.preprocessing import image
from keras.models import load_model
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image

# Function to calculate Structural Similarity Index (SSI)
def calculate_ssim(img_path, reference_image_paths):
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img = img.resize(Image.open(reference_image_paths[0]).size)  # Resize to match the dimensions of the first reference image

    img_array = np.array(img)
    
    ssim_values = [ssim(img_array, np.array(Image.open(reference_img).convert('L'))) for reference_img in reference_image_paths]

    return max(ssim_values)  # Choose the maximum SSIM value among all reference images

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
    # Resize the image to match the expected input size of the model
    target_size = (150, 150)
    img = image.load_img(image_path, target_size=target_size)
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
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "bmp"])


    if uploaded_file is not None:
        # Define the paths to reference images for each defect type
        dataset_directory = 'NEU Metal Surface Defects Data/train'
        defect_folders = ['Crazing', 'Inclusion', 'Patches', 'Pitted', 'Rolled', 'Scratches']
        reference_image_paths = [os.path.join(dataset_directory, defect_folder, f) for defect_folder in defect_folders for f in os.listdir(os.path.join(dataset_directory, defect_folder)) if f.endswith('.jpg')][:5]  # Choose the first 5 images from each folder

        # Calculate SSIM with the set of reference images
        ssim_value = calculate_ssim(uploaded_file, reference_image_paths)

        st.write(f"Maximum SSIM with the reference images: {ssim_value}")

        # Set a threshold for SSIM
        ssim_threshold = 0.3

        if ssim_value >= ssim_threshold:
            # Continue with defect assessment

            # Load the model only if the SSIM condition is met
            model = load_mobilenet_model()

            if model is not None:
                # Make predictions
                prediction = predict_defect(uploaded_file, model)

                # Display the results
                st.subheader("Prediction Results:")
                for i, class_name in enumerate(classes):
                    st.write(f"{class_name}: {prediction[0][i]}")

                # Assess the highest probability predicted and print out the class
                max_prob_class = assess_defect(prediction[0], classes)
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
