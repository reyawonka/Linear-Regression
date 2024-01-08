import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

readdata = pd.read_csv('Lab9Data.csv', skiprows=2)
readdata.columns = readdata.iloc[0]
readdata = readdata[1:].drop(1).reset_index(drop=True)

label_encoder = LabelEncoder()
readdata['BreachType'] = label_encoder.fit_transform(readdata['BreachType'])
readdata['Country'] = label_encoder.fit_transform(readdata['Country'])

readdata['TotalAffected'] = pd.to_numeric(readdata['TotalAffected'], errors='coerce')
firsty = readdata['TotalAffected']
firsty.fillna(firsty.mean(), inplace=True)

firstx = readdata[['BreachType', 'Country']]

X_train, X_test, y_train, y_test = train_test_split(firstx, firsty, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred_test = model.predict(X_test)

print("Two Variables - R^2:", r2_score(y_test, y_pred_test))
print("Two Variables - RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))

test_sizes = [0.2, 0.3, 0.4, 0.5]
for test_size in test_sizes:
    X_train, X_test, y_train, y_test = train_test_split(firstx, firsty, test_size=test_size, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print(f"\nRMSE: {rmse} \nSize: {test_size}\nR^2: {r2}")

