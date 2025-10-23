import numpy as np
import matplotlib.pyplot as plt
example_array=np.array([1,2,3])
example_e=np.exp(example_array)
print(example_array)
print(example_e)
def sigmoid(x):
    return 1/(1+np.exp(-x))
new_array=np.arange(-10,10)
y=sigmoid(new_array)
np.set_printoptions(precision=3)
print(np.c_[new_array, y])
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(new_array, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
plt.show()



