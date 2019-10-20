#predicting whether a person will click on a ad or  not

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
df=pd.read_csv('advertising.csv')
df.head()


# Separating features and Target variable
X=df.drop(['Clicked on Ad','Ad Topic Line','Timestamp','Country','City'],axis=1)
y=df['Clicked on Ad']

#converting dataset into array
X=X.values
y=y.values


#Training and testing
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
clf=LogisticRegression()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

#Saving the model
with open('model','wb') as f:
    pickle.dump('model',f)






