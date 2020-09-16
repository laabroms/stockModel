from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import os


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense


header_list = ['Date', 'Open', 'High', 'Low', 'Close']

df = pd.read_csv("btc_bars2.csv", names=header_list)


df = df.sort_values('Date')

# Double check the result
df.head()


# plt.figure(figsize=(18, 9))
# plt.plot(range(df.shape[0]), (df['Low']+df['High'])/2.0)
# plt.xticks(range(0, df.shape[0], 500), df['Date'].loc[::500], rotation=45)
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Mid Price', fontsize=18)
# plt.show()




dataset_train = df[:1000]
dataset_test = df[1000:]


training_set = dataset_train.iloc[:, 1:2].values


dataset_train.head()


sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)


X_train = []
y_train = []
for i in range(60, 1000):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


model = Sequential()

model.add(LSTM(units=50, return_sequences=True,
               input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=100, batch_size=32)


real_stock_price = dataset_test.iloc[:, 1:2].values
print(real_stock_price)


dataset_total = pd.concat(
    (dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 500):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)



plt.plot(real_stock_price, color = 'black', label = 'BTC Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted BTC Stock Price')
plt.title('BTC Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('BTC Stock Price')
plt.legend()
plt.show()