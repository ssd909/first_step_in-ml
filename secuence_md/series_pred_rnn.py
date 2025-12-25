import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Input
time_series = np.arange(500)/1000
def create_data(time_series,steps=10):
    x=[]
    y=[]
    for i in range(len(time_series)-steps):
        x.append(time_series[i:i+steps])
        y.append(time_series[i+steps])
    return np.array(x),np.array(y)
x_train,y_train=create_data(time_series)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
model=Sequential([
    Input(shape=(10,1)),
    SimpleRNN(32,activation='tanh'),
    Dense(1)
])
model.compile(optimizer='adam',loss='mse')
model.fit(x_train, y_train, epochs=1000, batch_size=16)
x_test=np.array([499,500,501,502,503,504,505,506,507,508])/1000
x_test=x_test.reshape((1,10,1))
y_test=np.array([509])
pred=model.predict(x_test)
print(pred[0][0])