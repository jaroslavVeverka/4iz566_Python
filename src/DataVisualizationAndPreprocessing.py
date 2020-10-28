import pandas as pd
import matplotlib.pyplot as plt
import h5py

source_path = '../data/UniversalBank.csv'

# load csv dataset
data = pd.read_csv(source_path)

# basic understanding of data
print(f'[BASIC_UNDERSTANDING] Dimension of the dataset:\n', data.shape)
print(f'[BASIC_UNDERSTANDING] Data types for each column:\n', data.dtypes)
print(f'[BASIC_UNDERSTANDING] First five rows:\n', data.head())
print(f'[BASIC_UNDERSTANDING] Last five rows:\n', data.tail())
print(f'[BASIC_UNDERSTANDING] Column names:\n', data.columns.tolist())
data.apply(lambda x: print(f'Data distribution of columns\n', x.describe()))
print(f'[BASIC UNDERSTANDING] Data distribution:\n', data.apply(lambda x: x.describe()))

# rename name of variables with white space
data = data.rename(columns={'ZIP Code': 'ZIP_Code',
                            'Personal Loan': 'Personal_Loan',
                            'Securities Account': 'Securities_Account',
                            'CD Account': 'CD_Account'})

# check missing values
print(f'[MISSING_VALUES] Are in dataset any missing values:\n', data.isnull().values.any())
print(f'[MISSING_VALUES] Number of missing values:\n', data.isnull().sum().sum())
print(f'[MISSING_VALUES] Number of missing values after columns:\n', data.isnull().sum())

data = data.drop(columns=['ID', 'ZIP_Code'])

# creating of histograms for all variables
data.hist(bins=10)
plt.show()

# dichotomizaton of categorical variables
data = pd.get_dummies(data, columns=['Family', 'Education'], drop_first=True)

# # create sub-dataframe
# print(data.loc[:, ['CreditCard']])
# print(data[['CreditCard']])

# creating of conditional distribution of variables to the target variable
fig, axs = plt.subplots(nrows=4, ncols=4)
fig.suptitle('Conditional distribution of variables to the target variable', fontsize=16)

data.groupby('Personal_Loan').Age.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[0, 0], legend=True)
axs[0, 0].set(xlabel='Age', ylabel='Count')
data.groupby('Personal_Loan').Experience.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[0, 1], legend=True)
axs[0, 1].set(xlabel='Experience', ylabel='Count')
data.groupby('Personal_Loan').Income.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[0, 2], legend=True)
axs[0, 2].set(xlabel='Income', ylabel='Count')
data.groupby('Personal_Loan').Family_2.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[0, 3], legend=True)
axs[0, 3].set(xlabel='Family_2', ylabel='Count')
data.groupby('Personal_Loan').Family_3.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[1, 0], legend=True)
axs[1, 0].set(xlabel='Family_3', ylabel='Count')
data.groupby('Personal_Loan').Family_4.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[1, 1], legend=True)
axs[1, 1].set(xlabel='Family_4', ylabel='Count')
data.groupby('Personal_Loan').CCAvg.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[1, 2], legend=True)
axs[1, 2].set(xlabel='CCAvg', ylabel='Count')
data.groupby('Personal_Loan').Education_2.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[1, 3], legend=True)
axs[1, 3].set(xlabel='Education_2', ylabel='Count')
data.groupby('Personal_Loan').Education_3.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[2, 0], legend=True)
axs[2, 0].set(xlabel='Education_3', ylabel='Count')
data.groupby('Personal_Loan').Mortgage.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[2, 1], legend=True)
axs[2, 1].set(xlabel='Mortgage', ylabel='Count')
data.groupby('Personal_Loan').CreditCard.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[2, 2], legend=True)
axs[2, 2].set(xlabel='Personal_Loan', ylabel='Count')
data.groupby('Personal_Loan').Securities_Account.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[2, 3], legend=True)
axs[2, 3].set(xlabel='Securities_Account', ylabel='Count')
data.groupby('Personal_Loan').CD_Account.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[3, 0], legend=True)
axs[3, 0].set(xlabel='CD_Account', ylabel='Count')
data.groupby('Personal_Loan').Online.hist(bins=10, figsize=(15, 10), alpha=0.4, ax=axs[3, 1], legend=True, )
axs[3, 1].set(xlabel='Online', ylabel='Count')

plt.show()

# show distribution of target variable
data.hist(column = 'Personal_Loan', bins=2, legend=True)
plt.show()

print(f'[TARGET DISTRIBUTION] Number of target with value 0:\n', sum(data['Personal_Loan'] == 0))
print(f'[TARGET DISTRIBUTION] Number of target with value 1:\n', sum(data['Personal_Loan'] == 1))

# creating of Feature-Feature Relationships via scatter_matrix
from pandas.plotting import scatter_matrix
scatter_matrix(data, figsize=(30, 30), diagonal='kde')
plt.show()

# creating of correlation matrix
import seaborn as sns

corr_matrix = data.corr()
plt.subplots(figsize=(20,15))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm')
plt.title('Heatmap for the Dataset', fontsize = 20)

plt.show()

# save preprocessed data as csv
final_data_path = './../data/UniversalBank_preprocessed.csv'
data.to_csv(final_data_path, index=False)




