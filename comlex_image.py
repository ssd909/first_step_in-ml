import os

import tensorflow as tf
import numpy as np
from keras.src.metrics.accuracy_metrics import accuracy
data_dir=os.path.expanduser('~/Desktop/file3')

train_horse_dir=os.path.join(data_dir,'horse')
train_human_dir=os.path.join(data_dir,'human')

train_dataset=tf.keras.utils.image_dataset_from_directory(
    directory=data_dir,
    image_size=(300,300),
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

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(300,300,3)),
    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])
model.summary()
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy']
              )


model.fit(
    train_dataset_final,epochs=10,verbose=2
)


