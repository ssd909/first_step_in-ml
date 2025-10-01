import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])
model.compile(optimizer='sgd',loss='mean_squared_error')
xs = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], dtype=float)
ys = np.array([1,3,5,7,9,11,13,15,17,19,21,23,25,27,29], dtype=float)
model.fit(xs,ys,epochs=500)
pred_value=model.predict(np.array([16]))
print(pred_value)