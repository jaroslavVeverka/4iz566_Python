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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

# cnts_pipeline = Pipeline([
#     ('scale', StandardScaler())
# ])

preprocess_pipeline = ColumnTransformer([
    ('num', StandardScaler(), ['Age', 'Income', 'CCAvg', 'Mortgage'])
], remainder='passthrough')

pipeline = Pipeline([('preprocessing', preprocess_pipeline),
                     ('classifier', LogisticRegression(max_iter=10000))])

pipeline.fit(X_train, y_train)
print(pipeline.score(X_train, y_train))
y_pred = pipeline.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_pred, y_test, labels=[0, 1])
print(acc_score)
print(conf_matrix)
print(classification_report(y_test, y_pred))
