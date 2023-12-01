import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

# Load your models
metal_classification_model = load_model('classified.h5')
defect_prediction_model = load_model('mobilenet_model (1).h5')

# Class labels for metal classification
metal_classes = ['plastic', 'paper', 'organic', 'metal', 'light blubs', 'glass', 'e-waste', 'clothes', 'batteries']

# Class labels for defect prediction
defect_classes = ['Pitted', 'Inclusion', 'Crazing', 'Patches', 'Scratches', 'Rolled']

def resize_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0


def metal_classification(img_array):
    metal_probs = metal_classification_model.predict(img_array)[0]
    metal_class = metal_classes[np.argmax(metal_probs)]
    return metal_class, metal_probs

def defect_prediction(img_array):
    defect_probs = defect_prediction_model.predict(img_array)[0]
    defect_class = defect_classes[np.argmax(defect_probs)]
    return defect_class, defect_probs

def main():
    st.title("Defects Assessment App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

        img_array = resize_image(uploaded_file)

        # Metal classification
        metal_class, metal_probs = metal_classification(img_array)

        if metal_class == 'metal':
            st.subheader("Metal Classification:")
            st.write(f"The uploaded image belongs to the class: {metal_class}")
            st.bar_chart({metal_classes[i]: metal_probs[i] for i in range(len(metal_classes))})

            # Defect prediction
            defect_class, defect_probs = defect_prediction(img_array)
            st.subheader("Defect Prediction:")
            st.write(f"The predicted defect class is: {defect_class}")
            st.bar_chart({defect_classes[i]: defect_probs[i] for i in range(len(defect_classes))})
        else:
            st.subheader("Metal Classification:")
            st.warning(f"The uploaded image does not belong to the metal class. Please upload a metal image.")

if __name__ == "__main__":
    main()
