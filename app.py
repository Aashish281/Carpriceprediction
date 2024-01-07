import pickle
from flask import Flask, request, jsonify, render_template

import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    input_data = np.array(list(data.values())).reshape(1, -1)
    output = model.predict(input_data)
    print(output[0])
    return jsonify({'prediction': output[0]})

if __name__ == "__main__":
    app.run(debug=True)

