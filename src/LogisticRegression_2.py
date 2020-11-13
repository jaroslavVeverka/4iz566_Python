import pandas as pd

from src.RandomForest import X_train

data_path = './../data/UniversalBank_Final.csv'
data = pd.read_csv(data_path)

# dependent and independents variables
y = data['Personal_Loan']
X = data.drop(columns=['Personal_Loan'])

print(f'Number of X: ', X.shape)
print(data.head())

print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 0:\n', sum(y == 0))
print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 1:\n', sum(y == 1))

print(f'[TARGET DISTRIBUTION] Nddddumber of target in test dataset with value 0:\n', sum(y == 0))
print(f'[TARGET DISTRIBUTION] Number of target in test dataset with value 1:\n', sum(y == 1))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

#
# BASIC MODEL
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# cnts_pipeline = Pipeline([
#     ('scale', StandardScaler())
# ])

num_atributes = ['Age', 'Income', 'CCAvg', 'Mortgage']
preprocessing_step = ColumnTransformer([
    ('num', StandardScaler(), num_atributes)
], remainder='passthrough')

steps = [
    ('preprocessing', preprocessing_step),
    ('classifier', LogisticRegression(max_iter=10000))
]

pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)

print(pipeline.score(X_train, y_train))
y_pred = pipeline.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_pred, y_test, labels=[0, 1])
print(acc_score)
print(conf_matrix)
print(classification_report(y_test, y_pred))

#
# HYPERPARAMER TUNING MODEL
#
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
import time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

start_time = time.time()
preprocess_pipeline = ColumnTransformer([
    ('num', StandardScaler(), ['Age', 'Income', 'CCAvg', 'Mortgage'])
], remainder='passthrough')

pipeline = Pipeline([('preprocessing', preprocess_pipeline),
                     ('classifier', LogisticRegression())])

params_space = {
    'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'classifier__penalty': ['l2'],
    'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

gs_logit = GridSearchCV(estimator=pipeline,
                        param_grid=params_space,
                        scoring='neg_root_mean_squared_error',
                        cv=10,
                        verbose=0)

gs_logit = gs_logit.fit(X_train, y_train)

print(gs_logit.best_estimator_)
print(gs_logit.best_score_)

for i in range(len(gs_logit.cv_results_['params'])):
    print(gs_logit.cv_results_['params'][i], 'test acc.:', gs_logit.cv_results_['mean_test_score'][i])

y_pred = gs_logit.best_estimator_.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_pred, y_test, labels=[0, 1])
print(acc_score)
print(conf_matrix)
print(classification_report(y_test, y_pred))

print(round(time.time() - start_time, 2))

#
# FEATURE SELECTION AND HYPERPARAMER TUNING MODEL
#
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
import time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

start_time = time.time()
preprocess_pipeline = ColumnTransformer([
    ('num', StandardScaler(), ['Age', 'Income', 'CCAvg', 'Mortgage'])
], remainder='passthrough')

pipeline = Pipeline([('preprocessing', preprocess_pipeline),
                     ('selector', SelectKBest()),
                     ('classifier', LogisticRegression())])

params_space = {
    'selector__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear'],
    'classifier__penalty': ['l2'],
    'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

gs_logit = GridSearchCV(estimator=pipeline,
                        param_grid=params_space,
                        scoring='neg_root_mean_squared_error',
                        cv=10,
                        verbose=0)

gs_logit = gs_logit.fit(X_train, y_train)

print(gs_logit.best_estimator_)
print(gs_logit.best_score_)

for i in range(len(gs_logit.cv_results_['params'])):
    print(gs_logit.cv_results_['params'][i], 'test acc.:', gs_logit.cv_results_['mean_test_score'][i])

y_pred = gs_logit.best_estimator_.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_pred, y_test, labels=[0, 1])
print(acc_score)
print(conf_matrix)
print(classification_report(y_test, y_pred))

print(round(time.time() - start_time, 2))

# #
# # FEATURE SELECTION AND HYPERPARAMER TUNING MODEL
# #
# from sklearn.feature_selection import SelectKBest
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# import time
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
# print('ahoj')
#
# start_time = time.time()
# preprocess_pipeline = ColumnTransformer([
#     ('num', StandardScaler(), ['Age', 'Income', 'CCAvg', 'Mortgage'])
# ], remainder='passthrough')
#
# pipeline = Pipeline([('preprocessing', preprocess_pipeline),
#                      ('selector', SelectKBest(k=5)),
#                      ('classifier',LogisticRegression())])
#
# params_space = [{'selector__k': [3, 4]},
#                 {'classifier': [LogisticRegression()],
#                  'classifier__solver': ['newton-cg', 'lbfgs', 'liblinear'],
#                  'classifier__penalty': ['l2'],
#                  'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
#                 ]
#
# gs_logit2 = GridSearchCV(pipeline, params_space, cv=10, verbose=0, scoring='neg_root_mean_squared_error')
# gs_logit2 = gs_logit2.fit(X_train, y_train)
#
# print(gs_logit2.best_estimator_)
# print(gs_logit2.best_score_)
#
# for i in range(len(gs_logit2.cv_results_['params'])):
#     print(gs_logit2.cv_results_['params'][i], 'test acc.:', gs_logit2.cv_results_['mean_test_score'][i])
#
# y_pred = gs_logit2.best_estimator_.predict(X_test)
#
# acc_score = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_pred, y_test, labels=[0, 1])
# print(acc_score)
# print(conf_matrix)
# print(classification_report(y_test, y_pred))
#
# print(round(time.time() - start_time, 2))
