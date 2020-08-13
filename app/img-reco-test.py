#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 23:03:13 2020

@author: lucas
"""

# for generating test images
# from PIL import Image
# from keras.datasets import mnist
# import numpy as np

# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# for i in np.random.randint(0, 10000+1, 10):
#     arr2im = Image.fromarray(X_train[i])
#     arr2im.save('./test/{}.png'.format(i), "PNG")

from keras.models import load_model
from PIL import Image
import numpy as np
from flasgger import Swagger

from flask import Flask, request
app = Flask(__name__)
swagger = Swagger(app)

model = load_model('model.h5')

@app.route('/predict-digit', methods=['POST'])
def predict_digit():
    """Example endpoint returning a prediction of mnist
    ---
    parameters:
        - name: image
          in: formData
          type: file
          required: true
    """
    im = Image.open(request.files['image'])
    im2arr = np.array(im).reshape((1, 1, 28, 28))
    return str(np.argmax(model.predict(im2arr)))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
    
    
    
    
    
    
    
    
    
