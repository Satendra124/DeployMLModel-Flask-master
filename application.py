import numpy as np
from flask import Flask, request, render_template
from keras.models import load_model
from PIL import Image
from numpy import asarray
import os
application = Flask(__name__) #Initialize the flask App

model = load_model('model.h5')

@application.route('/')
def home():
    return render_template('index.html')

class_names = ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy']

@application.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    print(request.files)
    f = request.files['file']
    f.save(f.filename)  
    # Import the necessary libraries


    # load the image and convert into
    # numpy array
    img = Image.open(f.filename)
    img = img.resize((256,256))

    # asarray() class is used to convert
    # PIL images into NumPy arrays
    numpydata = asarray(img)

    img_batch = np.expand_dims(numpydata, 0)

    predictions = model.predict(img_batch)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    print(predicted_class,confidence)
    res = {
        predicted_class:predicted_class,
        confidence:confidence
    }
    os.remove(f.filename)
    return render_template('predict.html', prediction_text='Prediction is :'+predicted_class)

if __name__ == "__main__":
    application.run(debug=True)
