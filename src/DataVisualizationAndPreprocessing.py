import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# nacteni vychoziho datasetu
source_path = '../data/UniversalBank.csv'
data = pd.read_csv(source_path)

# basic understanding of data
print('[BASIC_UNDERSTANDING] Dimension of the dataset:\n', data.shape)
print('[BASIC_UNDERSTANDING] Data types for each column:\n', data.dtypes)
print('[BASIC_UNDERSTANDING] First five rows:\n', data.head())
print('[BASIC_UNDERSTANDING] Last five rows:\n', data.tail())
print('[BASIC_UNDERSTANDING] Column names:\n', data.columns.tolist())
data.apply(lambda x: print('Data distribution of columns\n', x.describe()))
print('[BASIC UNDERSTANDING] Data distribution:\n', data.describe())

# check missing values
print(f'[MISSING_VALUES] Are in dataset any missing values:\n', data.isnull().values.any())
print(f'[MISSING_VALUES] Number of missing values:\n', data.isnull().sum().sum())
print(f'[MISSING_VALUES] Number of missing values after columns:\n', data.isnull().sum())

# rename name of variables with white space
data = data.rename(columns={'ZIP Code': 'ZIP_Code',
                            'Personal Loan': 'Personal_Loan',
                            'Securities Account': 'Securities_Account',
                            'CD Account': 'CD_Account'})

# drop unuseful variables
data = data.drop(columns=['ID', 'ZIP_Code'])

# creating of histograms for all variables
data.hist(bins=10)
plt.show()

# dichotomizaton of categorical variables
data = pd.get_dummies(data, columns=['Family', 'Education'], drop_first=True)

# creating of conditional distribution of variables to the target variable
fig, axs = plt.subplots(nrows=7, ncols=2)

data.groupby('Personal_Loan').Age.hist(bins=10, figsize=(15, 20), alpha=0.4, ax=axs[0, 0], legend=True)
axs[0, 0].set(xlabel='Age', ylabel='Count')
data.groupby('Personal_Loan').Experience.hist(bins=10, figsize=(15, 20), alpha=0.4, ax=axs[0, 1], legend=True)
axs[0, 1].set(xlabel='Experience', ylabel='Count')
data.groupby('Personal_Loan').Income.hist(bins=10, figsize=(15, 20), alpha=0.4, ax=axs[1, 0], legend=True)
axs[1, 0].set(xlabel='Income', ylabel='Count')
data.groupby('Personal_Loan').Family_2.hist(bins=2, figsize=(15, 20), alpha=0.4, ax=axs[1, 1], legend=True)
axs[1, 1].set(xlabel='Family_2', ylabel='Count')
data.groupby('Personal_Loan').Family_3.hist(bins=2, figsize=(15, 20), alpha=0.4, ax=axs[2, 0], legend=True)
axs[2, 0].set(xlabel='Family_3', ylabel='Count')
data.groupby('Personal_Loan').Family_4.hist(bins=2, figsize=(15, 20), alpha=0.4, ax=axs[2, 1], legend=True)
axs[2, 1].set(xlabel='Family_4', ylabel='Count')
data.groupby('Personal_Loan').CCAvg.hist(bins=10, figsize=(15, 20), alpha=0.4, ax=axs[3, 0], legend=True)
axs[3, 0].set(xlabel='CCAvg', ylabel='Count')
data.groupby('Personal_Loan').Education_2.hist(bins=2, figsize=(15, 20), alpha=0.4, ax=axs[3, 1], legend=True)
axs[3, 1].set(xlabel='Education_2', ylabel='Count')
data.groupby('Personal_Loan').Education_3.hist(bins=2, figsize=(15, 20), alpha=0.4, ax=axs[4, 0], legend=True)
axs[4, 0].set(xlabel='Education_3', ylabel='Count')
data.groupby('Personal_Loan').Mortgage.hist(bins=10, figsize=(15, 20), alpha=0.4, ax=axs[4, 1], legend=True)
axs[4, 1].set(xlabel='Mortgage', ylabel='Count')
data.groupby('Personal_Loan').CreditCard.hist(bins=2, figsize=(15, 20), alpha=0.4, ax=axs[5, 0], legend=True)
axs[5, 0].set(xlabel='CreditCard', ylabel='Count')
data.groupby('Personal_Loan').Securities_Account.hist(bins=2, figsize=(15, 20), alpha=0.4, ax=axs[5, 1], legend=True)
axs[5, 1].set(xlabel='Securities_Account', ylabel='Count')
data.groupby('Personal_Loan').CD_Account.hist(bins=2, figsize=(15, 20), alpha=0.4, ax=axs[6, 0], legend=True)
axs[6, 0].set(xlabel='CD_Account', ylabel='Count')
data.groupby('Personal_Loan').Online.hist(bins=2, figsize=(15, 20), alpha=0.4, ax=axs[6, 1], legend=True, )
axs[6, 1].set(xlabel='Online', ylabel='Count')

plt.show()

# show distribution of target variable
data.hist(column = 'Personal_Loan', bins=2, legend=True)
plt.show()

print(f'[TARGET DISTRIBUTION] Number of target with value 0:\n', sum(data['Personal_Loan'] == 0))
print(f'[TARGET DISTRIBUTION] Number of target with value 1:\n', sum(data['Personal_Loan'] == 1))

# creating of Feature-Feature Relationships via scatter_matrix

pd.plotting.scatter_matrix(data, c=data.loc[:, 'Personal_Loan'], figsize=(30, 30), diagonal='kde')
plt.show()

# creating of correlation matrix


corr_matrix = data.corr()
plt.subplots(figsize=(20,15))
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm')
plt.title('Heatmap for the Dataset', fontsize = 20)

plt.show()

# drop variable due to multicolinerity with Age
data = data.drop(columns='Experience')

# save preprocessed data as csv
data_path = './../data/UniversalBank_Final.csv'
data.to_csv(data_path, index=False)

# # dependent and independents variables
# Y = data['Personal_Loan']
# X = data.drop(columns=['Personal_Loan'])
#
# # split dataset into train and test part 8:2
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)
# print(f'Number of train X: ', X_train.shape)
# print(f'Number of test X: ', X_test.shape)
#
# # Applying of scaling on continuous variables
#
#
# scaler = StandardScaler()
# scaler.fit(X_train.loc[:, ['Age', 'Income', 'CCAvg', 'Mortgage']])
#
# X_train.loc[:, ['Age', 'Income', 'CCAvg', 'Mortgage']] = scaler.transform(X_train.loc[:,
#                                                                           ['Age', 'Income', 'CCAvg', 'Mortgage']])
# X_test.loc[:, ['Age', 'Income', 'CCAvg', 'Mortgage']] = scaler.transform(X_test.loc[:,
#                                                                          ['Age', 'Income', 'CCAvg', 'Mortgage']])
# #
# train_data = pd.concat([y_train, X_train], axis=1)
# test_data = pd.concat([y_test, X_test], axis=1)
# print(f'Number of train data: ', train_data.shape)
# print(f'Number of train data: ', test_data.shape)
#
# # save preprocessed data as csv
# train_data_path = './../data/UniversalBank_Train.csv'
# train_data.to_csv(train_data_path, index=False)
#
# # save preprocessed data as csv
# test_data_path = './../data/UniversalBank_Test.csv'
# test_data.to_csv(test_data_path, index=False)




