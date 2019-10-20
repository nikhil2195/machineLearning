import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle


df=pd.read_csv('train.csv')
df2=pd.read_csv('predict.csv')
df.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)
df2.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)


# Encoding the data into arbitrary numbers
df['Sex']=df['Sex'].astype("category").cat.codes
df['Embarked']=df['Embarked'].astype("category").cat.codes
df2['Sex']=df2['Sex'].astype("category").cat.codes
df2['Embarked']=df2['Embarked'].astype("category").cat.codes

# preprocessing the data
df.Age.fillna(df.Age.mean(),inplace=True)
df.Fare.fillna(df.Fare.mean(),inplace=True)
df2.Age.fillna(df2.Age.mean(),inplace=True)
df2.Fare.fillna(df2.Fare.mean(),inplace=True)


X=df.drop("Survived", axis=1)
y=df.Survived


# Training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

clf=LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)


# Saving the best fit model

with open('model','wb') as f:
    pickle.dump(clf,f)


clf.predict(df2)




