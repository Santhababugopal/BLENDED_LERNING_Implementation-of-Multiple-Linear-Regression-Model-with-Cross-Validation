# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start

2.Import Required Libraries

Import pandas and numpy for data handling

Import LinearRegression from sklearn.linear_model

Import cross_val_score and KFold from sklearn.model_selection

3.Load the Dataset

4.Read the car dataset using pandas.read_csv() (e.g., car_data.csv)

Preprocess the Data

5.Handle missing values if any

Select independent variables (X) — like engine size, mileage, year, etc.

6.Select dependent variable (y) — car price

Split the Data (optional)

You may split into training/testing sets for additional evaluation

Train the Multiple Linear Regression Model

7.Initialize the model: model = LinearRegression()

Fit the model using model.fit(X, y)

Evaluate Model Using Cross-Validation

Use cross_val_score() with KFold (e.g., cv=5)

Calculate and print the mean accuracy and standard deviation

8.End

## Program:
```
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
```
```
data = pd.read_csv('CarPrice_Assignment (1).csv')
```

```
data = data.drop(['car_ID','CarName'],axis=1)
data = pd.get_dummies(data,drop_first=True)

```

```
x = data.drop('price',axis=1)
y = data['price']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
```
```
model = LinearRegression()
model.fit(x_train,y_train)
```

```

print("\n===Cross Validation ===")
cv_scores = cross_val_score(model, x, y, cv=5)
print("Fold R² Scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R² Score: {cv_scores.mean():.4f}")

```

```
y_pred = model.predict(x_test)
print("\n===Test Set Perfomance ===")
print(f"MSE:{mean_squared_error(y_test,y_pred):.2f}")
print(f"R²: {r2_score(y_test,y_pred):.4f}")
```
```

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actua1 vs Predicted Prices")
plt .grid(True)
plt. show( )

```
## Output:
<img width="423" height="191" alt="image" src="https://github.com/user-attachments/assets/730c4129-e5e0-447c-afd4-55e1aeab153b" />
<img width="790" height="298" alt="image" src="https://github.com/user-attachments/assets/9a59e287-c4c3-4ef5-8fa3-e3dd72b2a92a" />
<img width="697" height="259" alt="image" src="https://github.com/user-attachments/assets/05377a91-32d3-4fe8-8027-39dbf34e482a" />
<img width="1001" height="647" alt="image" src="https://github.com/user-attachments/assets/ed545666-d2e1-4ca2-9b74-e08bb4ef98c5" />



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
