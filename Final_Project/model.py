from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import os
import json
import tweepy
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.stats import norm 

import pandas_datareader as pdr

import yfinance as yf
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)


app = Flask(__name__)
CORS(app)

global_df = None
global_ticker = None
global_sent_df = None


'''*******************************LSTM**********************************************'''

# Define an endpoint to handle prediction requests
def lstm_model(ticker):
    global global_df
    df = yf.download(tickers=[ticker], period='1y')
    y = df['Close'].ffill()
    y = y.values.reshape(-1, 1)

    # scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)

    # generate the input and output sequences
    n_lookback = 120  # length of input sequences (lookback period)
    n_forecast = 60  # length of output sequences (forecast period)

    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])

    X = np.array(X)
    Y = np.array(Y)

    # fit the model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, Y, epochs=100, batch_size=32, verbose=0)

    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    X_ = X_.reshape(1, n_lookback, 1)

    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)

    # organize the results in a data frame
    df_past = df[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past.at[df_past.index[-1], 'Forecast'] = df_past.at[df_past.index[-1], 'Actual']


    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan
    results = df_past._append(df_future).set_index('Date')
    global_df = results
    results = results.drop(results.index[:130])
    results_json = results.to_json(date_format='iso')
    return(results_json)




@app.route('/api/graph',methods=['GET','POST'])
def graph():
    if request.method == "POST":
        data = request.json
        ticker = data['ticker']
        df = lstm_model(ticker)
        #df_drop = df.drop(df.index[:200])
        return jsonify({'tested':df})

        

'''*******************************SENTIMENT**********************************************'''

def reddit_forecast(ticker):
    df = pd.read_csv(f"/Users/JerryStrippoli/Downloads/DS440/Final_Project/data/unique_data.csv")
    df = pd.DataFrame(df)
    global global_sent_df

    # Initialize the Sentiment Intensity Analyzer
    sia = SentimentIntensityAnalyzer()

    df['Title_Sentiment'] = df['Title'].astype(str).apply(lambda title: sia.polarity_scores(title)['compound'])

    # Convert 'Comments' to strings as well, to avoid TypeError with non-string types
    df['Comments_Sentiment'] = df['Comments'].astype(str).apply(lambda comments: sia.polarity_scores(comments)['compound'])
    # Display the updated DataFrame

    average_scores = df.groupby('ticker')[['Title_Sentiment', 'Comments_Sentiment']].mean().reset_index()

    # Display the resulting DataFrame
    #print(average_scores)

    # Normalize the average sentiment scores
    average_scores['Average_Sentiment'] = average_scores[['Title_Sentiment', 'Comments_Sentiment']].mean(axis=1)
    average_scores["norm_score"] = average_scores['Average_Sentiment'] - average_scores['Average_Sentiment'].min()
    average_scores["norm_score"] /= average_scores["norm_score"].max()

    yf.pdr_override()

    # Define the ticker and the start date
    start_date = "2023-1-1"
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date)

    # Create a DataFrame to hold the 'Adj Close' values
    data = pd.DataFrame(data['Close'], columns=['Close'])

    # Display the first few rows to verify
    #print(data.tail())

    # ticker = "AAPL"
    # data = pd.DataFrame()
    # data[ticker] = pdr.DataReader(ticker, data_source= yf, start="2023-1-1")['Adj Close']

    log_returns = np.log(1 + data.pct_change())
    u = log_returns.mean()
    var = log_returns.var()
    drift = u - (0.5 * var)
    stdev = log_returns.std()
    np.array(drift)
    x = np.random.rand(10,2)
    Z = norm.ppf(np.random.rand(10,2))
    t_intervals = 60
    iterations = 10
    daily_returns = np.exp(drift.values + stdev.values * norm.ppf(np.random.rand(t_intervals, iterations)))
    S0 = data.iloc[-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0
    for t in range(1, t_intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    sentiment_weight = average_scores.loc[average_scores['ticker'] == ticker, 'norm_score'].iloc[0]


    # Adjust drift and standard deviation according to the sentiment weight
    # This is a simple example; the exact formula should be based on your investment model
    adjusted_drift = drift.values * sentiment_weight
    adjusted_stdev = stdev.values * np.sqrt(sentiment_weight)  # This is an assumption; adjust as needed

    # Calculate daily returns using the adjusted drift and volatility
    daily_returns = np.exp(adjusted_drift + adjusted_stdev * norm.ppf(np.random.rand(t_intervals, iterations)))

    # Assuming S0 is the last adjusted close price
    S0 = data.iloc[-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = S0

    # Generate the price list for each day
    for t in range(1, t_intervals):
        price_list[t] = price_list[t - 1] * daily_returns[t]
    average_price_path = price_list.mean(axis=1)

    actual_data = data[-120:]

    # Last date from actual data
    last_actual_date = actual_data.index[-1]

    # Generate forecast dates starting from the day after the last actual date
    forecast_dates = pd.date_range(start=last_actual_date + pd.Timedelta(days=1), periods=60)

    # Assuming `forecasted_prices` is your 60-day forecast
    forecasted_prices = average_price_path

    forecast_series = pd.Series(forecasted_prices, index=forecast_dates)

    full_series = pd.concat([actual_data, forecast_series])
    results = full_series
    results.columns = ['Actual','Forecast']
    global_sent_df = results
    results_json = results.to_json(date_format='iso')
    return(results_json)





@app.route('/api/sentiment', methods = ["GET","POST"])

def sentiment():
    if request.method == "POST":
        data = request.json
        ticker = data['ticker']
        df = reddit_forecast(ticker)
        return jsonify({'sentiment_df':df})
    

 



'''*******************************%RETURN**********************************************'''


@app.route('/api/predict', methods=['GET','POST'])
def predict():
    global global_df  # Access the global variable within the function
    global global_sent_df
    if request.method == "POST":
        if global_df is not None and global_sent_df is not None:
            actual_values = global_df['Actual'].dropna().tolist()
            last_actual_value = actual_values[-1]
            last_forecast_value = global_df['Forecast'].dropna().iloc[-1]
            preturn = ((last_forecast_value - last_actual_value) / last_actual_value) * 100

            actual_values_sent = global_sent_df['Actual'].dropna().tolist()
            last_actual_value_sent = actual_values_sent[-1]
            last_forecast_value_sent = global_sent_df['Forecast'].dropna().iloc[-1]
            preturn_sent = ((last_forecast_value_sent - last_actual_value_sent) / last_actual_value_sent) * 100
            
            response_data = {'preturn': preturn, 'preturn_sent': preturn_sent}


            return jsonify({'PreturnDict': response_data})
        else:
            return jsonify({'error': 'DataFrame is not available'})








if __name__ == '__main__':
    app.run(debug=True)