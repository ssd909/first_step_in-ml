import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xs=np.array([1,2,3,4,5,6])
ys=np.array([1,3,5,7,9,11])
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer='sgd',loss='mse',metrics=['accuracy'])
history=model.fit(xs,ys,epochs=100)
pred_value=model.predict(np.array([7]))
history_loss=history.history['loss']
history_accuracy=history.history['accuracy']
epochs=range(1,len(history_loss)+1)
plt.figure(figsize=(12, 8))
plt.plot(epochs,history_loss,'r',label='Training loss')

plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.figure(figsize=(12, 8))
plt.plot(epochs,history_accuracy,'b',label='Training accuracy')
plt.title('Training  accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
