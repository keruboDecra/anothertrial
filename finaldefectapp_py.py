from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from keras.preprocessing import image
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the metal classification model
metal_model = load_model('classifyWaste.h5')

# Load the metal defect prediction model
defect_model = load_model('mobilenet_model.h5')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Check if the uploaded image is metal or not
        is_metal = predict_metal(file_path)

        if is_metal:
            # If it's metal, predict the defect
            defect_prediction = predict_defect(file_path)
            return render_template('result.html', result=f'This is a metal with defect: {defect_prediction}')
        else:
            return render_template('result.html', result='This is not a metal.')

    return redirect(request.url)

def predict_metal(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = metal_model.predict(img_array)
    return result[0][0] > 0.5  # Assuming a threshold of 0.5 for classification

def predict_defect(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    result = defect_model.predict(img_array)
    class_index = np.argmax(result)
    
    # Assuming your classes are in the following order
    classes = ['Scratches', 'Patches', 'Rolled', 'Pitted', 'Inclusion', 'Crazing']
    predicted_class = classes[class_index]

    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)
