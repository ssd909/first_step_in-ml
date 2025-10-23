import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.svm import SVC

data={
    'weight' :[100,120,90,230,140,234,312,450,367,98],
    'brightness':[2,8,3,5,7,9,10,4,1,6],
     'round':[2,8,3,5,7,9,10,4,1,6],

    'target':['apple','orange','orange','apple','orange','apple','orange','orange','apple','apple'],

}
data_form=pd.DataFrame(data=data)
data_form.to_csv('datas.csv',index=False)
loaded_data=pd.read_csv('data.csv')
loaded_data['label']=loaded_data['target'].map({'apple':1,'orange':0})
loaded_data.drop('target',axis=1,inplace=True)
print(loaded_data)

scaling_features=['brightness','round','weight']
ar=loaded_data[scaling_features]
scaler=MinMaxScaler()

x_scaled=scaler.fit_transform(ar)
loaded_data[scaling_features]=x_scaled

y=loaded_data['label']
x=loaded_data.drop('label',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
svm_model=SVC()
test_grib={
    'C' :Real(1e-2,1e3,prior='log-uniform'),
    'kernel':Categorical(['linear','rbf','sigmoid']),
    'gamma':Real(1e-2,1e3,prior='log-uniform')
}
bayes_search=BayesSearchCV (
    estimator=svm_model,
    search_spaces=test_grib,
    n_iter=50,
    cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=42),
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
bayes_search.fit(x_train,y_train)
print(bayes_search.best_params_)
print(bayes_search.best_score_)



