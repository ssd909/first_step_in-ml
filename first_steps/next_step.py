import tensorflow as tf
import numpy as np
import unittest

def create_training_data():
    feature =np.array([1,2,3,4,5,6,],dtype=float)
    target= np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
    return feature, target

features, targets = create_training_data()
print(features.shape,targets.shape)
def define_model_and_compile():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='sgd',loss='mse')
    return model

untrained_model = define_model_and_compile()
untrained_model.summary()
def train_model():
    x,y = create_training_data()
    model = define_model_and_compile()
    model.fit(x,y,epochs=500)
    return model
model = train_model()
pred_room=np.array([7.0])
pred_price=model.predict(pred_room)

print(round(pred_price[0][0],3))


