
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
x_tr_scaled=StandardScaler().fit_transform(x_train)
x_tst_scaled=StandardScaler().fit_transform(x_test)
knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_tr_scaled,y_train)

plt.figure(figsize=(10,10))
plt.scatter(
    x_tr_scaled[y_train==0,0],
    x_tr_scaled[y_train==0,1],
    c='red',

)
plt.scatter(
    x_tr_scaled[y_train==1,0],
    x_tr_scaled[y_train==1,1],
    c='yellow',

)
plt.scatter(
    x_tr_scaled[y_train==0,2],
    x_tr_scaled[y_train==0,3],
    c='blue',
)
plt.scatter(
    x_tr_scaled[y_train==1,2],
    x_tr_scaled[y_train==1,3],
    c='black',
)


plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['y1','y2'])


plt.show()