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

# dependent and independents variables
Y = data['CreditCard']
X = data.drop(columns=['ID', 'CreditCard'])

#
# Sequential Forward Selection - Logistic regression
#
from sklearn import linear_model
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)
print(f'Number of train X: ', X_train.shape)
print(f'Number of test X: ', X_test.shape)

logit = linear_model.LogisticRegression(max_iter = 1000)

# forward step, 10-folds cross validation
sfs1 = SFS(logit,
           k_features=(2, 10),
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)

sfs1 = sfs1.fit(X, Y)
# print best 5 predictors
print(sfs1.k_feature_names_)
# accuracy score of model in 10-folds cross validation
print(sfs1.k_score_)

print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
print('all subsets:\n', sfs1.subsets_)
plot_sfs(sfs1.get_metric_dict(), kind='std_err')
plt.ylim([0.7, 0.8])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()

# keep only 5 best predictors
X_train_sfs = sfs1.transform(X_train)
X_test_sfs = sfs1.transform(X_test)
print(f'Number of train X sfs: ', X_train_sfs.shape)
print(f'Number of test X sfs: ', X_test_sfs.shape)

# estimate Logit
logit = linear_model.LogisticRegression()
logit.fit(X_train_sfs, y_train)
print(logit.coef_)

# calculate probabilities
predicted_proba = logit.predict_proba(X_test_sfs)
predicted_proba = predicted_proba[:, 1]

# calculate AUC score - how good is model ( 0.5 - randomness, 1 - best model)
auc_value = roc_auc_score(y_test, predicted_proba)
print(round(auc_value,2))

# confusion matrix and accuracy on test data
predicted_target = logit.predict(X_test_sfs)
print(accuracy_score(predicted_target, y_test))
print(confusion_matrix(predicted_target, y_test, labels=[0, 1]))



