import pandas as pd

source_path = '../data/carInsurance.csv'

# load csv dataset
dataFile = pd.read_csv(source_path)

# convert dataFile to Pandas DataFrame
data = pd.DataFrame(dataFile)

# print first 10 rows and names of columns
print(data.head(10))
print(list(data.columns))