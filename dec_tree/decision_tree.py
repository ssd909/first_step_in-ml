from sklearn.tree import DecisionTreeClassifier,plot_tree
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
iris=load_iris()

x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

tree_model=DecisionTreeClassifier(criterion='gini',random_state=42)
tree_model.fit(x_train,y_train)
accuracy=tree_model.score(x_test,y_test)
plt.figure(figsize=[15,10])
plot_tree(tree_model, feature_names=iris.feature_names, class_names=iris.target_names,rounded=True,filled=True)
plt.show()

