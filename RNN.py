import pandas as pd
import numpy as np
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, LSTM, Input
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def create_window(data, window_size, features):
    window_data = np.array([])
    for i in range(data.shape[0] - window_size):
        window_data = np.append(window_data, data[i:i + window_size, :])
    window_data = window_data.reshape(-1, window_size, features)
    return window_data


window_size = 7
train = pd.read_csv('data/HYMTF.csv', sep=',')

x_data = train[['Close', 'Volume', 'Adj Close']].to_numpy()

scaler = MinMaxScaler()
x_scaled = scaler.fit(x_data)
x_data = x_scaled.fit_transform(x_data)

x_data = create_window(x_data, window_size, features=3)

train_target = train.shift(-window_size)
y_data = train_target['Close'][:-window_size].values.reshape(-1, 1)
y_scaled = scaler.fit(y_data)
y_data = y_scaled.fit_transform(y_data)
'''
return_sequences=True : ,x.shape==y.shape
return_sequences=False : x.shape=3,y.shape=2
'''
inputs = Input(batch_shape=(1, x_data.shape[1], x_data.shape[2]))
lstm = LSTM(6, return_sequences=True, stateful=True)(inputs)
lstm = LSTM(12, stateful=True)(lstm)
outputs = Dense(1, activation='sigmoid')(lstm)
model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])
model.fit(x_data, y_data, epochs=5)

y_data = y_scaled.inverse_transform(y_data)
predict = model.predict(x_data, verbose=1)
predict = y_scaled.inverse_transform(predict)

plt.plot(y_data[:500, :], color='B')
plt.plot(predict[:500, :], color='R')
