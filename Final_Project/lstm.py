import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import LSTM

 
# Load multiple stock datasets
stocks = ['AAPL', 'MSFT'] 
models = {}

for ticker in stocks:
    # Load dataset
    df = pd.read_csv(f"/Users/JerryStrippoli/Downloads/DS440/FrontEnd/data/{ticker}.csv", na_values=['null'], index_col='Date', parse_dates=True, infer_datetime_format=True)

    output_var = pd.DataFrame(df['Adj Close'])

    # Selecting the Features
    features = ['Open', 'High', 'Low', 'Volume']

    scaler = MinMaxScaler()
    feature_transform = scaler.fit_transform(df[features])
    feature_transform = pd.DataFrame(columns=features, data=feature_transform, index=df.index)

    timesplit = TimeSeriesSplit(n_splits=10)
    for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()

    # Process the data for LSTM
    trainX = np.array(X_train)
    testX = np.array(X_test)
    X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Building LSTM Model
    lstm = Sequential()
    lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))
    lstm.add(Dense(1))
    lstm.compile(loss='mean_squared_error', optimizer='adam')

    # Training the LSTM model
    history = lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

    # Save the trained model
    model_filename = f"pretrained_lstm_model_{ticker}.h5"
    lstm.save(model_filename)
    models[ticker] = model_filename

    # Predicting using the trained model
    y_pred = lstm.predict(X_test)
    prediction_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred.reshape(-1)})
    prediction_df.to_csv('/Users/JerryStrippoli/Downloads/DS440/FrontEnd/data/predict_{}.csv'.format(ticker), index=False)
