import os
from flask import Flask, request, redirect, url_for, render_template
import numpy as np
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import load_model
from skimage.transform import resize
import tensorflow as tf
from tf import keras
import cv2

app = Flask(__name__)

# loading model
print("Loading model")
my_model = keras.models.load_model('my_model')
my_model.load_weights("weights.h5")

@app.route('/', methods=['GET', 'POST'])
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('home.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    #Step 1
    test_data = []
    img_dims = 150
    img = plt.imread(os.path.join('uploads', filename))
    img = cv2.resize(img, (img_dims, img_dims))
    img = np.dstack([img, img, img])
    img = img.astype('float32') / 255
    test_data.append(img)
    test_data = np.array(test_data)
    
    #Step 2
    probabilities = my_model.predict(test_data)[0][0]
    print("the possibility is", probabilities)
    
    #Step 3
    accuracy = probabilities*100
    print("the accuracy is", accuracy)
    # result = ""
    if accuracy <= 49:
        result = "normal"
    elif accuracy >= 50:
        result = "positive"  
    #Step 4
    return render_template('after.html', result=result)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)