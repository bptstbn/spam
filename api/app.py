from flask import Flask, jsonify, request
from tensorflow import keras
import joblib
import json
import pandas as pd
import numpy as np

#create an instance of Flask
app = Flask(__name__)


model = keras.models.load_model('assets/model.h5')
transformer = joblib.load('assets/transformer.joblib')


@app.route('/', methods=['GET', 'POST'])
def predict():
    string = request.args.get('input')
    data = pd.DataFrame.from_dict(json.loads(string))
    probabilities = model.predict(transformer.transform(data))
    predictions = np.where(probabilities > 0.5, 'spam', 'not spam')
    return jsonify(predictions.tolist())


if __name__ == '__main__':
    app.run(debug=True)