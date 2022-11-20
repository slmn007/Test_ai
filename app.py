from flask import Flask, render_template, request

import numpy as np

from tensorflow.keras.models import load_model

app = Flask(__name__)

model_path = "model/faceClassification3.h5"
model_train = load_model(model_path)

@app.route('/', methods=['GET'])
def index():
    
    return render_template('index.html')

if __name__ == '__main__':
    
    app.run()