import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

readdata = pd.read_csv('Lab9Data.csv', skiprows=2)
readdata.columns = readdata.iloc[0]
readdata = readdata[1:].drop(1).reset_index(drop=True)

readdata['TotalAffected'] = pd.to_numeric(readdata['TotalAffected'], errors='coerce')


readdata['TotalAffected'].plot(kind='hist')
plt.title('Histogram of Total Affected')
plt.xlabel('Total Affected')
plt.ylabel('Frequency')
plt.show()

readdata['BreachType'].value_counts().plot(kind='barh')
plt.title('Breach Types Count')
plt.xlabel('Count')
plt.ylabel('Breach Type')
plt.show()

readdata['Country'].value_counts().plot(kind='bar')
plt.title('Number of Breaches per Country')
plt.xlabel('Country')
plt.ylabel('Breaches Count')
plt.show()


label_encoder = LabelEncoder()
readdata['BreachType'] = label_encoder.fit_transform(readdata['BreachType'])

