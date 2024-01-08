import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

readdata = pd.read_csv('Lab9Data.csv', skiprows=2)
readdata.columns = readdata.iloc[0]
readdata = readdata[1:].drop(1).reset_index(drop=True)

label_encoder = LabelEncoder()
readdata['BreachType'] = label_encoder.fit_transform(readdata['BreachType'])

readdata['TotalAffected'] = pd.to_numeric(readdata['TotalAffected'], errors='coerce')
firsty = readdata['TotalAffected']
firsty.fillna(firsty.mean(), inplace=True)

firstx = readdata[['BreachType']] 

X_train, X_test, y_train, y_test = train_test_split(firstx, firsty, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Coefficient of determination (R^2):", r2_score(y_test, y_pred_test))
print("Root Mean Square Error (RMSE):", np.sqrt(mean_squared_error(y_test, y_pred_test)))
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, color='blue', label='Predicted')
plt.scatter(y_test, y_test, color='red', label='Actual')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()
