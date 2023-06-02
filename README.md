# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
  1.Prepare your data
    -Collect and clean data on employee salaries and features
    -Split data into training and testing sets
    
    

  2.Define your model
    -Use a Decision Tree Regressor to recursively partition data based on input features
    -Determine maximum depth of tree and other hyperparameters
    

  3.Train your model
    -Fit model to training data
    -Calculate mean salary value for each subset
    

  4.Evaluate your model
    -Use model to make predictions on testing data
    -Calculate metrics such as MAE and MSE to evaluate performance
    

  5.Tune hyperparameters
    -Experiment with different hyperparameters to improve performance
    

  6.Deploy your model
    Use model to make predictions on new data in real-world application.
    

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Praveen s
RegisterNumber:  212222240077
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
### Initial dataset:
![O1](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393556/764ddfe2-dede-4e3e-ba55-15f2b4354e05)
### Data Info:
![O2](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393556/f3fbe7ba-f0c2-4eb1-8ff1-941a57dcb5ec)
### Optimization of null values:
![O3](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393556/7bdb6adb-95ec-41e9-8230-a794023c7670)
### Converting string literals to numericl values using label encoder:
![O4](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393556/d39011ae-e3e6-4bc4-bd15-a832eb8b7e21)
### Assigning x and y values:
![O5](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393556/5b458c10-e726-4cdc-8c60-fc69b7a3c314)
### Mean Squared Error:
![O6](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393556/3dee0095-d3a1-4f6e-a74e-b6f3983ec5e3)
### R2 (variance):
![O7](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393556/924accc7-ece9-491e-9a8c-f980aae9da81)
### Prediction:
![O8](https://github.com/LATHIKESHWARAN/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/119393556/d4d9f997-f736-42c0-963b-acd94b10d596)
## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
