import pandas as pd

data_path = './../data/UniversalBank_Final.csv'
data = pd.read_csv(data_path)

# dependent and independents variables
y = data['Personal_Loan']
X = data.drop(columns=['Personal_Loan'])

print(f'Number of X: ', X.shape)
print(data.head())

print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 0:\n', sum(y == 0))
print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 1:\n', sum(y == 1))

print(f'[TARGET DISTRIBUTION] Number of target in test dataset with value 0:\n', sum(y == 0))
print(f'[TARGET DISTRIBUTION] Number of target in test dataset with value 1:\n', sum(y == 1))

from sklearn.neighbors import KNeighborsClassifier
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

preprocess_pipeline = ColumnTransformer([
    ('num', StandardScaler(), ['Age', 'Income', 'CCAvg', 'Mortgage'])
], remainder='passthrough')

pipeline = Pipeline([('preprocessing', preprocess_pipeline),
                     ('classifier', KNeighborsClassifier())])

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
from sklearn.model_selection import GridSearchCV
import time

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

start_time = time.time()
preprocess_pipeline = ColumnTransformer([
    ('num', StandardScaler(), ['Age', 'Income', 'CCAvg', 'Mortgage'])
], remainder='passthrough')

pipeline = Pipeline([('preprocessing', preprocess_pipeline),
                     ('classifier', KNeighborsClassifier())])

params_space = {
    'classifier__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
}

gs_logit = GridSearchCV(estimator=pipeline,
                        param_grid=params_space,
                        scoring='accuracy',
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
# HYPERPARAMER TUNING MODEL AND FEATURE SELECTION
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
                     ('classifier', KNeighborsClassifier())])

params_space = {
    'selector__k': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'classifier__n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'classifier__metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
}

gs_logit = GridSearchCV(estimator=pipeline,
                        param_grid=params_space,
                        scoring='accuracy',
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
