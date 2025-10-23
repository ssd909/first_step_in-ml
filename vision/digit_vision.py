import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
data_i=tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=data_i.load_data()
index=10
np.set_printoptions(linewidth=200)
print(train_images[index])
print(train_labels[index])
plt.imshow(train_images[index])
plt.colorbar()
plt.show()