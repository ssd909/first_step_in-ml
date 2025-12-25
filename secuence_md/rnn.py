import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN,Input

def cr_dat(length=1000,max_len=10):
  x=np.random.randint(0,2,(length,max_len,1))
  y=np.sum(x,axis=1)%2
  return x,y
x_train,y_train=cr_dat()
model=Sequential([
    Input(shape=(10,1)),
    SimpleRNN(2,activation='tanh'),
    Dense(1,activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)