import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# loading data from csv file
data_path = './../data/UniversalBank_preprocessed.csv'
data = pd.read_csv(data_path)

print(data.head())

# dependent and independents variables
Y = data['Personal_Loan']
X = data.drop(columns=['Personal_Loan'])

# split dataset into train and test part 8:2
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)
print(f'Number of train X: ', X_train.shape)
print(f'Number of test X: ', X_test.shape)

print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 0:\n', sum(y_train == 0))
print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 1:\n', sum(y_train == 1))

print(f'[TARGET DISTRIBUTION] Number of target in test dataset with value 0:\n', sum(y_test == 0))
print(f'[TARGET DISTRIBUTION] Number of target in test dataset with value 1:\n', sum(y_test == 1))

#
# basic model of Random forest
#

rf_0 = RandomForestClassifier( random_state=1)
rf_0.fit(X_train, y_train)

# testing
predicted_target = rf_0.predict(X_test)
predicted_proba = rf_0.predict_proba(X_test)
predicted_proba_target = predicted_proba[:, 1]

# evaluation
auc_value_0 = roc_auc_score(y_test, predicted_proba_target)
print('[RF_0] AUC value of model:\n', auc_value_0)

acc_score_0 = accuracy_score(y_test, predicted_target)
conf_matrix_0 = confusion_matrix(predicted_target, y_test, labels=[0, 1])
print(acc_score_0)
print(conf_matrix_0)
print(classification_report(y_test, predicted_target))

import matplotlib.pyplot as plt
import scikitplot as skplt

skplt.metrics.plot_cumulative_gain(y_test, predicted_proba)
plt.show()

skplt.metrics.plot_lift_curve(y_test, predicted_proba)
plt.show()

from sklearn import tree
plt.figure(figsize=(20,20))
_ = tree.plot_tree(rf_0.estimators_[0], feature_names=X.columns, filled=True)
plt.show()

#
# basic model of Random forest with feature selection
#

rf_1 = RandomForestClassifier(random_state=1)

sfs1 = SFS(rf_1,
           k_features=(1, 13),
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)

sfs1 = sfs1.fit(X_train, y_train)

print(sfs1.k_feature_names_)
# accuracy score of model in 10-folds cross validation
print(sfs1.k_score_)

print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
print('all subsets:\n', sfs1.subsets_)
plot_sfs(sfs1.get_metric_dict(), kind='std_err')
plt.ylim([0.2, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()

# keep only 5 best predictors
X_train_sfs = sfs1.transform(X_train)
X_test_sfs = sfs1.transform(X_test)
print(f'Number of train X sfs: ', X_train_sfs.shape)
print(f'Number of test X sfs: ', X_test_sfs.shape)

# training
rf_1 = RandomForestClassifier(random_state=1)
rf_1.fit(X_train_sfs, y_train)

# testing
predicted_target = rf_1.predict(X_test_sfs)
predicted_proba = rf_1.predict_proba(X_test_sfs)
predicted_proba_target = predicted_proba[:, 1]

# evaluation
auc_value_1 = roc_auc_score(y_test, predicted_proba_target)
print('[RF_1] AUC value of model:\n', auc_value_1)

acc_score_1 = accuracy_score(y_test, predicted_target)
conf_matrix_1 = confusion_matrix(predicted_target, y_test, labels=[0, 1])
print(acc_score_1)
print(conf_matrix_1)
print(classification_report(y_test, predicted_target))

import matplotlib.pyplot as plt
import scikitplot as skplt

skplt.metrics.plot_cumulative_gain(y_test, predicted_proba)
plt.show()

skplt.metrics.plot_lift_curve(y_test, predicted_proba)
plt.show()

from sklearn import tree
plt.figure(figsize=(20,20))
_ = tree.plot_tree(rf_1.estimators_[0], feature_names=X.columns, filled=True)
plt.show()
