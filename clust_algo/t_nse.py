import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
iris=load_iris()
x=iris.data
dt=pd.DataFrame(x,columns=iris.feature_names)

x_scaled=StandardScaler().fit_transform(dt)

t_nse=TSNE(n_components=2,
           perplexity=100,
           learning_rate='auto',
           max_iter=1000,
           random_state=42
           )
x_ember=t_nse.fit_transform(x_scaled)
plt.figure(figsize=(10,10))
plt.scatter(x_ember[:,0],
            x_ember[:,1],)
plt.xlabel('x')
plt.ylabel('y')
plt.show()



