# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

 
  1. Import pandas, numpy and sklearn
  2. Calculate the values for the training data set
  3. Calculate the values for the test data set
  4. Plot the graph for both the data sets and calculate for MAE, MSE and RMSE

 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Yamunaasri
RegisterNumber:  212222240117
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/student_scores.csv')
#Displying the contents in datafile
df.head()

df.tail()

#Segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,-1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying the predicted values
Y_pred

#displaying the actual values
Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="green")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE= ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print("MAE= ",mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:


### df.head()
![output](mll%201.png)

### df.tail()
![output](mll%202.png)

### X
![output](mll%203.png)

### Y
![output](mll%204.png)

### PREDICTED Y VALUES
![output](mll%205.png)

### ACTUAL Y VALUES
![output](mll%206.png)

### GRAPH FOR TRAINING DATA
![output](mll%207.png)

### GRAPH FOR TEST DATA
![output](mll%208.png)

### MEAN SQUARE ERROR, MEAN ABSOLUTE ERROR AND RMSE
![output](mll%209.png)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
