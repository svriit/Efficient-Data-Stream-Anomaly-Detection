import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dropout, RepeatVector, TimeDistributed, Dense
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go

# Streamlit app setup
st.title("Stock Price Anomaly Detection with LSTM Autoencoder")

# User inputs for ticker
ticker = st.text_input("Enter the stock ticker symbol (e.g., 'GE')", 'GE')
start_date = st.text_input("Enter the start date (YYYY-MM-DD)", '2000-01-01')
end_date = st.text_input("Enter the end date (YYYY-MM-DD)", pd.Timestamp.today().strftime('%Y-%m-%d'))
trigger = st.button("Run Analysis")

if ticker and trigger:
    # Fetch live data from Yahoo Finance
    dataframe = yf.download(ticker, start=start_date, end=end_date)
    df = dataframe.reset_index()[['Date', 'Close']]

    # Plot the data
    df['Date'] = pd.to_datetime(df['Date'])
    st.line_chart(df.set_index('Date')['Close'])

    st.write("Start date is:", df['Date'].min())
    st.write("End date is:", df['Date'].max())

    # Split the data into train and test sets
    if len(df) < 2:
        st.error("Not enough data to split into training and test sets. Please enter a larger date range.")
    else:
        train, test = df.loc[df['Date'] <= '2003-12-31'], df.loc[df['Date'] > '2003-12-31']

    # Normalize the dataset using StandardScaler
        if len(train) > 0:
            scaler = StandardScaler()
            scaler = scaler.fit(train[['Close']])
            train['Close'] = scaler.transform(train[['Close']])
            test['Close'] = scaler.transform(test[['Close']])
        else:
            st.error("Training set is empty after split. Please adjust the date range.")

    # Function to convert the dataset to sequences
    def to_sequences(x, y, seq_size=1):
        x_values = []
        y_values = []
        for i in range(len(x) - seq_size):
            x_values.append(x.iloc[i:(i + seq_size)].values)
            y_values.append(y.iloc[i + seq_size])
        return np.array(x_values), np.array(y_values)

    # Define the sequence size
    seq_size = 30

    # Convert the training and test data into sequences
    trainX, trainY = to_sequences(train[['Close']], train['Close'], seq_size)
    testX, testY = to_sequences(test[['Close']], test['Close'], seq_size)

    # Define the LSTM autoencoder model
    model = Sequential()
    model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
    model.add(Dropout(rate=0.2))
    model.add(RepeatVector(trainX.shape[1]))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(rate=0.2))
    model.add(TimeDistributed(Dense(trainX.shape[2])))
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    # Train the model
    with st.spinner('Training the model...'):
        history = model.fit(trainX, trainY, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

    # Plot training and validation loss
    st.line_chart(pd.DataFrame({"Training Loss": history.history['loss'], "Validation Loss": history.history['val_loss']}))

    # Predict on train and test data
    with st.spinner('Making predictions...'):
        trainPredict = model.predict(trainX)
    trainMAE = np.mean(np.abs(trainPredict - trainX), axis=1)
    st.write("Training MAE Histogram")
    st.bar_chart(np.histogram(trainMAE, bins=30)[0])

    # Define threshold for anomaly detection
    max_trainMAE = np.percentile(trainMAE, 95)  # Use 95th percentile as threshold

    with st.spinner('Predicting on test data...'):
        testPredict = model.predict(testX)
    testMAE = np.mean(np.abs(testPredict - testX), axis=1)
    st.write("Test MAE Histogram")
    st.bar_chart(np.histogram(testMAE, bins=30)[0])

    # Capture all details in a DataFrame for easy plotting
    anomaly_df = pd.DataFrame(test[seq_size:])
    anomaly_df['testMAE'] = testMAE
    anomaly_df['max_trainMAE'] = max_trainMAE
    anomaly_df['anomaly'] = anomaly_df['testMAE'] > anomaly_df['max_trainMAE']
    anomaly_df['Close'] = test[seq_size:]['Close']

    # Plot testMAE vs max_trainMAE
    st.line_chart(anomaly_df.set_index('Date')[['testMAE', 'max_trainMAE']])

    # Extract anomalies
    anomalies = anomaly_df.loc[anomaly_df['anomaly'] == True]

    # Plot anomalies and close price in a combined chart
    st.write("Close Price and Anomalies")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(anomaly_df['Date'], anomaly_df['Close'], label='Close Price', color='blue')
    ax.scatter(anomalies['Date'], anomalies['Close'], color='red', label='Anomalies', marker='o')
    ax.set_xlabel('Day of the Month')
    ax.set_ylabel('Close Price')
    ax.legend()
    st.pyplot(fig, use_container_width=True)

    # Determine buy, sell, or hold signal based on anomalies and display it as a Plotly gauge
    if anomalies.empty:
        signal = "HOLD"
        meter_value = 50
    else:
        last_anomaly_date = anomalies['Date'].iloc[-1]
        last_close_price = scaler.inverse_transform([[anomaly_df.loc[anomaly_df['Date'] == last_anomaly_date, 'Close'].values[0]]])[0][0]
        current_close_price = scaler.inverse_transform([[test['Close'].iloc[-1]]])[0][0]

        if current_close_price > last_close_price * 1.05:
            signal = "STRONG BUY"
            meter_value = 100
        elif current_close_price < last_close_price * 0.95:
            signal = "STRONG SELL"
            meter_value = 0
        else:
            signal = "HOLD"
            meter_value = 50

    st.write(f"Signal: {signal}")

    # Display the signal as a Plotly gauge divided into three parts
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=meter_value,
        title={'text': "Signal Meter"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 33], 'color': "red"},
                {'range': [33, 66], 'color': "yellow"},
                {'range': [66, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': meter_value
            }
        }
    ))
    st.plotly_chart(fig)

    # Extract and display the latest 10 news items
    st.write("News")
    import requests
    from bs4 import BeautifulSoup

    news_url = f'https://finance.yahoo.com/quote/{ticker}/'
    response = requests.get(news_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        news_sections = soup.find_all('section', class_='container sz-small block yf-1sxfjua responsive hideImageSmScreen', limit=10)
        news_data = []
        for section in news_sections:
            link_tag = section.find('a', class_='subtle-link fin-size-small titles noUnderline yf-1e4diqp')
            if link_tag:
                news_title = link_tag['title']
                news_url = link_tag['href']
                news_data.append({'Title': news_title, 'URL': f'https://finance.yahoo.com{news_url}'})
        if news_data:
            news_df = pd.DataFrame(news_data)
            st.write(news_df)
        else:
            st.write("No news items found.")
    else:
        st.write("Failed to retrieve news data.")