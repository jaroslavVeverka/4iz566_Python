import pandas as pd
import matplotlib.pyplot as plt

source_path = '../data/UniversalBank.csv'

# load csv dataset
data = pd.read_csv(source_path)

# any missing values?
print(data.isnull().sum())
print(data.head(10))

# list of variable names
print(list(data.columns))

# create sub-dataframe
print(data.loc[:, ['CreditCard']])
print(data[['CreditCard']])

# exploring of dependent variable
rows_number = len(data)
target_count = sum(data['CreditCard'] == 1)
print(target_count/rows_number)

plt.hist(data['CreditCard'], bins=2, rwidth=0.85)

plt.title('Histogram of Credit Card variable')
plt.xlabel('CreditCard')
plt.ylabel('Count')
plt.xticks([0.25,0.75], ['no', 'yes'])
plt.grid(axis='y')
plt.show()
plt.clf()





