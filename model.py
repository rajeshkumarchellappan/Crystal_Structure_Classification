import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df=pd.read_csv('classification_crystal_structure.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)
print(df.head())

X=df.drop('Crystal_Structure',axis=1)
y=df['Crystal_Structure']
print(X.columns)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=20)

#Preparing Random Forest Classification
rfc_model=RandomForestClassifier()
rfc_model.fit(X_train.values,y_train.values)
y_pred_rfc=rfc_model.predict(X_test.values)
print(len(X_train.columns))
print(len(X_test.columns))
import pickle
# # Saving model to disk
pickle.dump(rfc_model, open('model.pkl','wb'),protocol=pickle.HIGHEST_PROTOCOL)
model=pickle.load(open('model.pkl','rb'))
print(y_pred_rfc)
