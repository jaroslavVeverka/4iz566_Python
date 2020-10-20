import pandas as pd
import matplotlib.pyplot as plt

source_path = '../data/UniversalBank.csv'

# load csv dataset
dataFile = pd.read_csv(source_path)

# convert dataFile to Pandas DataFrame
data = pd.DataFrame(dataFile)

print(data.isnull().sum())
print(data.head(10))
print(list(data.columns))

#some plots
plt.hist(data['CreditCard'], bins=2, rwidth=0.85)

plt.title('Histogram of Credit Card variable')
plt.xlabel('CreditCard')
plt.ylabel('Count')
plt.xticks([0,1], ['no', 'yes'])
plt.grid(axis='y')
plt.show()
plt.clf()




