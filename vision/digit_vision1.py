import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
data_i=tf.keras.datasets.mnist
(train_images,train_labels),(test_images,test_labels)=data_i.load_data()
index=0
fig,ax=plt.subplots(
    nrows=2,
    ncols=2,
    figsize=(8, 8),

)
ax_flat=ax.flatten()
for i in range(2):
 ax_flat[i].imshow(train_images[i])
plt.show() 