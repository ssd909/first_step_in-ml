import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
iris=load_iris()
X=iris.data

dt=pd.DataFrame(X,columns=iris.feature_names)
x_scaled=StandardScaler().fit_transform(dt)
pca=PCA(n_components=2)
x_pc=pca.fit_transform(x_scaled)
pca_frame=pd.DataFrame(x_pc,columns=['PC1','PC2'])
print(pca.explained_variance_ratio_.sum())
print(pca.components_)
plt.figure(figsize=(10,10))
plt.scatter(
    x_pc[:,0],
    x_pc[:,1],
    alpha=0.5,
)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(['PC1','PC2'])
plt.grid(True)
plt.show()

