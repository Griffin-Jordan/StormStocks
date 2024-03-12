from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

app = Flask(__name__)
CORS(app)

# Function to preprocess input data
def preprocess_input_data(data):
    features = ['Open', 'High', 'Low', 'Volume']
    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(data[features])
    feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=data.index)
    return np.array(feature_transform).reshape(-1, 1, len(features))

# Define an endpoint to handle prediction requests
@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data['ticker']
    model_filename = f'pretrained_lstm_model_{ticker}.h5'  # Adjust the filename based on the ticker
    model = load_model(model_filename)
    df = pd.read_csv(f'data/{ticker}.csv')  # Adjust the path to your CSV files
    input_data = preprocess_input_data(df)
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)