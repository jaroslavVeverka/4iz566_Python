import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn import linear_model
from sklearn.ensemble._gradient_boosting import np_bool
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# # loading data from csv file
train_data_path = './../data/UniversalBank_Train.csv'
test_data_path = './../data/UniversalBank_Test.csv'
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(test_data_path)

# dependent and independents variables
y_train = train_data['Personal_Loan']
y_test = test_data['Personal_Loan']
X_train = train_data.drop(columns=['Personal_Loan'])
X_test = test_data.drop(columns=['Personal_Loan'])

print(f'Number of train X: ', X_train.shape)
print(f'Number of test X: ', X_test.shape)

print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 0:\n', sum(y_train == 0))
print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 1:\n', sum(y_train == 1))

print(f'[TARGET DISTRIBUTION] Number of target in test dataset with value 0:\n', sum(y_test == 0))
print(f'[TARGET DISTRIBUTION] Number of target in test dataset with value 1:\n', sum(y_test == 1))

#
# basic model of logistic regression
#

# training
logit_0 = linear_model.LogisticRegression(max_iter=10000)
logit_0.fit(X_train, y_train)
print('[LOGIT_0] Estimated coefficient:\n', logit_0.coef_)

# testing
predicted_target = logit_0.predict(X_test)
predicted_proba = logit_0.predict_proba(X_test)
predicted_proba_target = predicted_proba[:, 1]

# evaluation
auc_value_0 = roc_auc_score(y_test, predicted_proba_target)
print('[LOGIT_0] AUC value of model:\n', auc_value_0)

acc_score_0 = accuracy_score(y_test, predicted_target)
conf_matrix_0 = confusion_matrix(predicted_target, y_test, labels=[0, 1])
print(acc_score_0)
print(conf_matrix_0)
print(classification_report(y_test, predicted_target))

# import matplotlib.pyplot as plt
# import scikitplot as skplt
#
# skplt.metrics.plot_cumulative_gain(y_test, predicted_proba)
# plt.show()
#
# skplt.metrics.plot_lift_curve(y_test, predicted_proba)
# plt.show()
#
# #
# #  model of logistic regression with selected features via Forward stepwise selection and balanced target
# #
#
# # Balancing training dataset with SMOTE method using oversampling
# # from imblearn.over_sampling import SMOTE
# #
# # X_train, y_train = SMOTE().fit_resample(X_train, y_train)
# # print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 0:\n', sum(y_train == 0))
# # print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 1:\n', sum(y_train == 1))
#
# logit_1 = linear_model.LogisticRegression(max_iter=10000)
#
# sfs1 = SFS(logit_1,
#            k_features=(1, 13),
#            forward=True,
#            floating=False,
#            verbose=2,
#            scoring='neg_root_mean_squared_error',
#            cv=10)
#
# sfs1 = sfs1.fit(X_train, y_train)
#
# print(sfs1.k_feature_names_)
# # accuracy score of model in 10-folds cross validation
# print(sfs1.k_score_)
#
# print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
# print('all subsets:\n', sfs1.subsets_)
# plot_sfs(sfs1.get_metric_dict(), kind='std_err')
# plt.ylim([0.2, 1])
# plt.title('Sequential Forward Selection (w. StdDev)')
# plt.grid()
# plt.show()
#
# # keep only 5 best predictors
# X_train_sfs = sfs1.transform(X_train)
# X_test_sfs = sfs1.transform(X_test)
# print(f'Number of train X sfs: ', X_train_sfs.shape)
# print(f'Number of test X sfs: ', X_test_sfs.shape)
#
# # training
# logit_1 = linear_model.LogisticRegression(max_iter=10000)
# logit_1.fit(X_train_sfs, y_train)
# print('[LOGIT_1] Estimated coefficient:\n', logit_1.coef_)
#
# # testing
# predicted_target = logit_1.predict(X_test_sfs)
# predicted_proba = logit_1.predict_proba(X_test_sfs)
# predicted_proba_target = predicted_proba[:, 1]
#
# # evaluation
# auc_value_1 = roc_auc_score(y_test, predicted_proba_target)
# print('[LOGIT_1] AUC value of model:\n', auc_value_1)
#
# acc_score_1 = accuracy_score(y_test, predicted_target)
# conf_matrix_1 = confusion_matrix(predicted_target, y_test, labels=[0, 1])
# print(acc_score_1)
# print(conf_matrix_1)
#
# # cumulative gains curve
# import matplotlib.pyplot as plt
# import scikitplot as skplt
#
# # 20 % of best probas contain 25 % of all targets
# skplt.metrics.plot_cumulative_gain(y_test, predicted_proba)
# plt.show()
#
# # lift curve
# skplt.metrics.plot_lift_curve(y_test, predicted_proba)
# plt.show()
#
#
# #basic model of logistic regression - metaparameter tuning
#
# from sklearn.model_selection import GridSearchCV
#
# param_grid = {
#     'solver': ['newton-cg', 'lbfgs', 'liblinear'],
#     'penalty': ['l2'],
#     'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
# }
#
# logit_cv_grid = GridSearchCV(estimator=linear_model.LogisticRegression(max_iter=10000),
#                              param_grid=param_grid,
#                              cv=10,
#                              scoring='neg_root_mean_squared_error')
#
# logit_cv_grid.fit(X_train_sfs, y_train)
# print(logit_cv_grid.best_score_, logit_cv_grid.best_params_)
#
# logit_3 = logit_cv_grid.best_estimator_
# #logit_3.fit(X_train, y_train)
# print('[LOGIT_3] Estimated coefficient:\n', logit_3.coef_)
#
# # testing
# predicted_target = logit_3.predict(X_test_sfs)
# predicted_proba = logit_3.predict_proba(X_test_sfs)
# predicted_proba_target = predicted_proba[:, 1]
#
# # evaluation
# auc_value_3 = roc_auc_score(y_test, predicted_proba_target)
# print('[LOGIT_3] AUC value of model:\n', auc_value_3)
#
# acc_score_3 = accuracy_score(y_test, predicted_target)
# conf_matrix_3 = confusion_matrix(predicted_target, y_test, labels=[0, 1])
# print(acc_score_3)
# print(conf_matrix_3)
# print(classification_report(y_test, predicted_target))
#
# import matplotlib.pyplot as plt
# import scikitplot as skplt
#
# skplt.metrics.plot_cumulative_gain(y_test, predicted_proba)
# plt.show()
#
# skplt.metrics.plot_lift_curve(y_test, predicted_proba)
# plt.show()


# logit = linear_model.LogisticRegression(max_iter = 10000)
#
# # forward step, 10-folds cross validation
# # pro vyber adekvatnich promennych je pouzit Forward step selection v ramci nehoz
# # dochazi s postupnemu odhadu modelu s ruznymi promenymi a jejich kombinacemi na trenovacicha datech. Promenne
# # jsou postupne vybirany na zaklade hodnoty spravnosti klasifikace daneho modelu  v ramci cross validace, tj.
# # na 1/10 testovacich dat trenovaciho datasetu.
# # V ramci prvni iterace se pro n promenych odhadne a v ramci cross validace vyhodoti n modelu.
# # V ramci iterace se vyvere ten model Mmax1 = max_acc(Mi), jehoz predikcini schopnosti v ramci cross validace jsou nejvyssi.
# # V ramci 2. iterace se pro n-1 promennych odhadne a v ramci cross validace vhodnoti n-1 modelu, jez jsou specialnejsi verzi
# # modelu Mmax1. y = x1
# # V ramci iterace se vyvere ten model Mmax2, jehoz predikcini schopnosti v ramci cross validace jsou nejvyssi.
# # Toto se opakuje do doby kdy neni v modelu pocet pozadovanych promennych, nebo do doby, kdy je pocet
# # promenych v pozadovanem intervalu a dalsi iterace neprinasi zlepseni predikcnich schopnosti.


from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection  import SelectKBest, f_classif

logit_4 = linear_model.LogisticRegression(max_iter=10000)

# sfs2 = SFS(estimator=logit_4,
#            k_features=(1,13),
#            forward=True,
#            floating=False,
#            scoring='neg_root_mean_squared_error',
#            cv=3)

pipe = Pipeline([('selector', SelectKBest(f_classif, k=1)),
                 ('classifier', logit_4)])

param_grid = {
    'selector__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'classifier__penalty': ['l2'],
    'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

gs = GridSearchCV(estimator=pipe,
                  param_grid=param_grid,
                  scoring='neg_root_mean_squared_error',
                  n_jobs=1,
                  cv=10,
                  iid=True)

# run gridearch
gs = gs.fit(X_train, y_train)

for i in range(len(gs.cv_results_['params'])):
    print(gs.cv_results_['params'][i], 'test acc.:', gs.cv_results_['mean_test_score'][i])


print("Best parameters via GridSearch", gs.best_params_)

logit_4 = gs.best_estimator_
#logit_3.fit(X_train, y_train)
print('[LOGIT_3] Estimated coefficient:\n', logit_4)

# testing
predicted_target = logit_4.predict(X_test)
predicted_proba = logit_4.predict_proba(X_test)
predicted_proba_target = predicted_proba[:, 1]

# evaluation
auc_value_4 = roc_auc_score(y_test, predicted_proba_target)
print('[LOGIT_3] AUC value of model:\n', auc_value_4)

acc_score_4 = accuracy_score(y_test, predicted_target)
conf_matrix_4 = confusion_matrix(predicted_target, y_test, labels=[0, 1])
print(acc_score_4)
print(conf_matrix_4)
print(classification_report(y_test, predicted_target))