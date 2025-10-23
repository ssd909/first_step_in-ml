import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import numpy as np
from tensorflow.python.keras.saving.saved_model.serialized_attributes import metrics

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
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

""
""
svm_model=SVC()
test_grib={
    'C':[0.1,1,10,50,100,1000],
    'gamma':[0.01,0.02,0.03,0.04,0.05],
    'kernel':['linear','rbf','sigmoid'],
    'degree':[2,3,4,5,6,7],



}
grib_search=GridSearchCV(
    estimator=svm_model,
    param_grid=test_grib,
    scoring='accuracy',
    cv=4,
    verbose=2,
)
grib_search.fit(x_train,y_train)
print(grib_search.best_params_)
print(grib_search.best_score_)
