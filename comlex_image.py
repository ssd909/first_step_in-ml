import os

import tensorflow as tf
import numpy as np
from keras.src.metrics.accuracy_metrics import accuracy
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir=os.path.expanduser('~/Desktop/file3')
data_dir_val=os.path.expanduser('~/Desktop/file4')
train_horse_dir=os.path.join(data_dir,'horse')
train_human_dir=os.path.join(data_dir,'human')

train_dataset=tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    image_size=(150,150),
    batch_size=5,
    label_mode='binary'
)
train_dataset_val=tf.keras.utils.image_dataset_from_directory(
    directory=data_dir_val,
    image_size=(150,150),
    batch_size=5,
    label_mode='binary'
)
rescale_layer=tf.keras.layers.Rescaling(1./255)
train_dataset_rescale=train_dataset.map(lambda image, label: (rescale_layer(image), label))
train_dataset_final=(
    train_dataset_rescale
    .cache()
    .shuffle(buffer_size=len(train_dataset))
    .prefetch(tf.data.experimental.AUTOTUNE)
)
rescale_layer_val=tf.keras.layers.Rescaling(1./255)
train_dataset_rescale_val=train_dataset_val.map(lambda image, label: (rescale_layer_val(image), label))
train_dataset_final_val=(
    train_dataset_rescale_val
    .cache()
    .shuffle(buffer_size=len(train_dataset_val))
    .prefetch(tf.data.experimental.AUTOTUNE)
)


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(150,150,3)),
    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy']
              )


history=model.fit(

    train_dataset_final,epochs=10,verbose=2,
    validation_data=train_dataset_rescale_val
)
acc=history.history['accuracy']
acc_val=history.history['val_accuracy']
epochs=range(len(acc))
plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, acc_val, 'r', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc=0)
plt.show()

