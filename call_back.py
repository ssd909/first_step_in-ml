import tensorflow as tf
import numpy as np
from keras.src.callbacks import callback
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy')>0.5:
            print("model has converged")
            self.model.stop_training=True

callbacks=MyCallback()
store_data=tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels)=store_data.load_data()
train_images=train_images/255.1
test_images=test_images/255.1

model=tf.keras.models.Sequential([
    tf.keras.Input(shape=(28,28,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),

])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5,callbacks=[callbacks])
