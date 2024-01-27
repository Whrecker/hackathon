import pandas as pd
import numpy as np
import random
day=[]
time=[]
accident=[]
count=0
#preparing sample data
for i in range(1000):
    day.append(random.choice(["rainy","sunny","cloudy"]))
    time.append(random.choice(["day","night"]))
    accident.append(random.choice([True,False]))
    dic={"day type":day,"time":time,"accident":accident}
df=pd.DataFrame(dic)
#wrangling data
accidents=pd.DataFrame({"accident":accident})
dayy=pd.get_dummies(df["day type"])
timee=pd.get_dummies(df["time"])
df=pd.concat([dayy,timee,accidents],axis=1)
print(df)
#finding possible accidents in future
X=df.drop("accident",axis=1)
y=df["accident"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=1)
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,y_train)
predictions= logmodel.predict(X_test)
print(predictions)

