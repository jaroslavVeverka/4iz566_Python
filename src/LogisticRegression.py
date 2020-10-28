import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# loading data from csv file
data_path = './../data/UniversalBank_preprocessed.csv'
data = pd.read_csv(data_path)

print(data.head())


# dependent and independents variables
Y = data['Personal_Loan']
X = data.drop(columns=['Personal_Loan'])

#
# Sequential Forward Selection - Logistic regression
#
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, stratify = Y)
print(f'Number of train X: ', X_train.shape)
print(f'Number of test X: ', X_test.shape)

logit = linear_model.LogisticRegression(max_iter = 10000)

# forward step, 10-folds cross validation
# pro vyber adekvatnich promennych je pouzit Forward step selection v ramci nehoz
# dochazi s postupnemu odhadu modelu s ruznymi promenymi a jejich kombinacemi na trenovacicha datech. Promenne
# jsou postupne vybirany na zaklade hodnoty spravnosti klasifikace daneho modelu  v ramci cross validace, tj.
# na 1/10 testovacich dat trenovaciho datasetu.
# V ramci prvni iterace se pro n promenych odhadne a v ramci cross validace vyhodoti n modelu.
# V ramci iterace se vyvere ten model Mmax1 = max_acc(Mi), jehoz predikcini schopnosti v ramci cross validace jsou nejvyssi.
# V ramci 2. iterace se pro n-1 promennych odhadne a v ramci cross validace vhodnoti n-1 modelu, jez jsou specialnejsi verzi
# modelu Mmax1. y = x1
# V ramci iterace se vyvere ten model Mmax2, jehoz predikcini schopnosti v ramci cross validace jsou nejvyssi.
# Toto se opakuje do doby kdy neni v modelu pocet pozadovanych promennych, nebo do doby, kdy je pocet
# promenych v pozadovanem intervalu a dalsi iterace neprinasi zlepseni predikcnich schopnosti.
sfs1 = SFS(logit,
           k_features=(2, 10),
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)

sfs1 = sfs1.fit(X_train, y_train)
# print best predictors
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

# estimate Logit
logit = linear_model.LogisticRegression(max_iter = 10000)
logit.fit(X_train_sfs, y_train)
print(logit.coef_)

# calculate probabilities
predicted_proba = logit.predict_proba(X_test_sfs)
predicted_proba_target = predicted_proba[:, 1]

# calculate AUC score - how good is model ( 0.5 - randomness, 1 - best model)
auc_value = roc_auc_score(y_test, predicted_proba_target)
print(round(auc_value,2))

# confusion matrix and accuracy on test data
predicted_target = logit.predict(X_test_sfs)
print(accuracy_score(predicted_target, y_test))
print(confusion_matrix(predicted_target, y_test, labels=[0, 1]))

# cumulative gains curve
import matplotlib.pyplot as plt
import scikitplot as skplt

# 20 % of best probas contain 25 % of all targets
skplt.metrics.plot_cumulative_gain(y_test, predicted_proba)
plt.show()

# lift curve
skplt.metrics.plot_lift_curve(y_test, predicted_proba)
plt.show()
