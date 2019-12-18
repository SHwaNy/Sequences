import pandas as pd
import numpy as np
from tensorflow.keras.layers import SimpleRNN,Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def create_window(data,window_size):
    window_data=[]
    for i in range(data.shape[0]-window_size+1):
        window_data.append(data[i:i+window_size,:])
    return np.array(window_data)
window_size=15
train=pd.read_csv('data/HYMTF.csv')

data=train['Close'].values.reshape(-1,1)

scaler=MinMaxScaler()
scaled=scaler.fit(data)
scaled_data=scaled.fit_transform(data)

data=create_window(scaled_data,window_size)
x_data=data[:,:window_size-1,:]
y_data=data[:,window_size-1:,:]
y_data=y_data.reshape(-1,1)
'''
return_sequences=True : ,x.shape==y.shape
return_sequences=False : x.shape=3,y.shape=2
'''
model=Sequential()
model.add(LSTM(64,batch_input_shape=(1,x_data.shape[1],x_data.shape[2]),return_sequences=True,stateful=True))
model.add(LSTM(128))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='mse',optimizer='adam',metrics=['acc'])
model.fit(x_data,y_data,epochs=5)

y_data=scaled.inverse_transform(y_data)
predict=model.predict(x_data)
predict=scaled.inverse_transform(predict)

plt.plot(y_data[:500,:],color='B')
plt.plot(predict[:500,:],color='R')



