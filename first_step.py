import tensorflow as tf
import numpy as np
xs=np.array([1,2,3,4,5,6])
ys=np.array([1,3,5,7,9,11])
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer='sgd',loss='mse')
model.fit(xs,ys,epochs=500)
pred_value=model.predict(np.array([7]))
print(pred_value)
