import pandas as pd

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
print(f'Number of test X: ', X_test.shape)
print(f'Number of test X: ', X_test.shape)

print(f'Number of train X: ', X_train.shape)
print(f'Number of test X: ', X_test.shape)
print(f'Number of test X: ', X_test.shape)
print(f'Number of test X: ', X_test.shape)

print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 0:\n', sum(y_train == 0))
print(f'[TARGET DISTRIBUTION] Number of target in train dataset with value 1:\n', sum(y_train == 1))

print(f'[TARGET DISTRIBUTION] Number of target in test dataset with value 0:\n', sum(y_test == 0))
print(f'[TARGET DISTRIBUTION] Number of target in test dataset with value 1:\n', sum(y_test == 1))

# Basic model KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

knn_0 = KNeighborsClassifier()

knn_0.fit(X_train, y_train)
print('Accuracy on train data: ', knn_0.score(X_train, y_train))

y_pred = knn_0.predict(X_test)
print('Accuracy on test data: ', accuracy_score(y_pred, y_test))
print('Accuracy on test data: ', knn_0.score(X_test, y_test))
print('Confusion matrix:\n', confusion_matrix(y_pred, y_test))

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
knn_1 = KNeighborsClassifier()

params = {
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # 'weights': ['uniform', 'distance'],
    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
}

gs_knn_1 = GridSearchCV(knn_1, param_grid=params, cv=10)

gs_knn_1.fit(X_train, y_train)

print("Best parameters via GridSearch", gs_knn_1.best_params_, 'ACC: ', gs_knn_1.best_score_)

knn_1 = gs_knn_1.best_estimator_

y_pred = knn_1.predict(X_test)
print('Accuracy on test data: ', accuracy_score(y_pred, y_test))
print('Accuracy on test data: ', knn_1.score(X_test, y_test))
print('Confusion matrix:\n', confusion_matrix(y_pred, y_test))



