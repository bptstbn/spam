from flask import Flask, jsonify, request
from tensorflow import keras
import joblib
import json
import pandas as pd
import numpy as np
from features_from_text import get_string


#create an instance of Flask
app = Flask(__name__)


# load pretrained model & transformer
model = keras.models.load_model('assets/model.h5')
transformer = joblib.load('assets/transformer.joblib')


# create the first route
# prediction from features
@app.route('/', methods=['GET', 'POST'])
def predict():
    string = request.args.get('input')
    data = pd.DataFrame.from_dict(json.loads(string))
    probabilities = model.predict(transformer.transform(data))
    predictions = np.where(probabilities > 0.5, 'spam', 'not spam')
    return jsonify(predictions.tolist())


# create the second route
# prediction from raw email text
@app.route('/fromtext', methods=['GET', 'POST'])
def predictfromtext():
    email = request.args.get('input')
    string = get_string(email)
    data = pd.DataFrame.from_dict(json.loads(string))
    probabilities = model.predict(transformer.transform(data))
    predictions = np.where(probabilities > 0.5, 'spam', 'not spam')
    return jsonify(predictions.tolist())


if __name__ == '__main__':
    app.run(debug=True)