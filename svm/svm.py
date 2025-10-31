import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np


data={
    'weight' :[100,120,90,230,140,234,312,450,367,98],
    'brightness':[2,8,3,5,7,9,10,4,1,6],

    'target':['apple','orange','orange','apple','orange','apple','orange','orange','apple','apple'],

}
table=pd.DataFrame(data=data)
table.to_csv('table_class.csv',index=False)
loaded_data=pd.read_csv('table_class.csv')
loaded_data['target_bool']=loaded_data['target'].map({'apple':1,'orange':0})
loaded_data.drop('target',axis=1,inplace=True)
""
#scaling
feature_scaling=['brightness','weight']
ar=loaded_data[feature_scaling]

scaler=MinMaxScaler()
x_scaled=scaler.fit_transform(ar)
scaled_data=pd.DataFrame(x_scaled,columns=feature_scaling)
loaded_data[feature_scaling]=scaled_data
print(loaded_data)
""
""
#data_split

y=loaded_data['target_bool']
x=loaded_data.drop('target_bool',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.5,random_state=42)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

""
""
#training svm model
svm_model=SVC(kernel='sigmoid',gamma=0.7784063026431617,C= 427.3620086107224,random_state=42)
svm_model.fit(x_train,y_train)
y_pred=svm_model.predict(x_test)
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))

""
""
#graphic
x_min, x_max = x_train.iloc[:, 0].min() - 0.1, x_train.iloc[:, 0].max() + 0.1
y_min, y_max = x_train.iloc[:, 1].min() - 0.1, x_train.iloc[:, 1].max() + 0.1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))
sd=np.c_[xx.ravel(), yy.ravel()]
sd_data=pd.DataFrame(sd,columns=x_train.columns)
Z = svm_model.predict(sd_data)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10, 7))

# გამყოფი საზღვრის (Decision Boundary) შევსება ფერით
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)

# --- 5. მონაცემთა წერტილების დახატვა ---
plt.scatter(
    x_train.iloc[:, 0], # X ღერძი (წონა)
    x_train.iloc[:, 1], # Y ღერძი (სიკაშკაშე)
    c=y_train, # ფერი კლასის მიხედვით
    cmap=plt.cm.coolwarm, # ფერების რუკა
    marker='o',
    edgecolor='k',
    s=70 # ზომა
)

plt.xlabel('weight')
plt.ylabel('brightness')


plt.show()


""