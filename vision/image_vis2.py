import tensorflow as tf
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
import matplotlib.pyplot as plt
fashion = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
def nn_mode(learning_rate=0.001):
 model = tf.keras.Sequential([
    tf.keras.Input(shape=(28,28,1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
 ])
 opti=tf.keras.optimizers.Adam(learning_rate=learning_rate)
 model.compile(optimizer=opti,loss='sparse_categorical_crossentropy',metrics=['accuracy'],)
 return model

par_grid={
    'model__learning_rate':[0.001,],
    'batch_size':[32,64],
    'epochs':[5,10],
}
cro_model=KerasClassifier(
    model=nn_mode,
    verbose=0
)
grid=GridSearchCV(estimator=cro_model,
                  param_grid=par_grid,
                  scoring='accuracy',

                  cv=5


                  )

grid.fit(train_images,train_labels)
print(grid.best_params_)

