import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from utils import create_sequences

def train_and_predict():

    df = pd.read_excel("data/company_cashflow.xlsx")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    if 'NetCash' not in df.columns:
        df['NetCash'] = df['CashIn'] - df['CashOut']

    df.set_index('Date', inplace=True)

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['NetCash']])

    SEQ = 12
    X, y = create_sequences(scaled_data, SEQ)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = X_train.reshape((X_train.shape[0], SEQ, 1))
    X_test = X_test.reshape((X_test.shape[0], SEQ, 1))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(SEQ, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1)

    model.save("models/lstm_model.h5")

    pred = model.predict(X_test)
    pred = scaler.inverse_transform(pred)
    y_test_actual = scaler.inverse_transform(y_test)

    dates_test = df.index[SEQ + split:]

    result = pd.DataFrame({
        "Date": dates_test,
        "Actual": y_test_actual.flatten(),
        "Forecast": pred.flatten()
    }).set_index("Date")

    result.to_json("outputs/forecast_output.json", orient="records", date_format="iso")

    return result
