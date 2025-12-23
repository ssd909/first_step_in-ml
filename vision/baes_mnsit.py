import tensorflow as tf
import keras
import keras_tuner as kt
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
fashion = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
def build_model(hp):
   model = keras.Sequential([])
   model.add(keras.layers.Flatten(input_shape=(28, 28,1)))

   hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
   model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))

   hp_dropout = hp.Float('dropout', min_value=0.0, max_value=1.0, step=0.1)
   model.add(tf.keras.layers.Dropout(hp_dropout))

   model.add(tf.keras.layers.Dense(10, activation='softmax'))
   hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
   model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),metrics=['accuracy'],loss='sparse_categorical_crossentropy')
   return model
tuner=kt.BayesianOptimization(
       hypermodel=build_model,
       max_trials=5,
       executions_per_trial=1,
      objective='val_accuracy'
   )
tuner.search(
    train_images,
    train_labels,
    epochs=5,
    validation_split=0.2,

)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.get)

