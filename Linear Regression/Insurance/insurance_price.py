import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pickle 
df=pd.read_csv("insurance.csv")
df.head()

#Preprocessiong the data, creating dummies 
sex=pd.get_dummies(df.sex)
regions=pd.get_dummies(df.region)
smoking=pd.get_dummies(df.smoker)
sex.drop(columns=['female'],inplace=True)
regions.drop(columns=['southwest'],inplace=True)
smoking.drop(columns=['no'],inplace=True)

#Combining the processed data and spitting them into target and features,target=X,features=y
merged=pd.concat([df,sex,smoking,regions],axis='columns')
merged.drop(columns=['sex','smoker','region'],inplace=True)
X=merged.drop(columns=['charges',])
y=merged.charges


#Splitting the target and features into test and train dataset
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2)
clf=linear_model.LinearRegression()
clf.fit(X_train,y_train)


#Dumping the trained model to use again
with open('model3','wb') as f:
    pickle.dump(lreg,f)
pickle_in=open('model','rb')
clf=pickle.load(pickle_in)
clf.score(X_test,y_test)




