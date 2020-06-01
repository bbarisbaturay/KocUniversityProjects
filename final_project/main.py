import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, validation_curve, GridSearchCV
from math import sqrt
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from yellowbrick.model_selection import ValidationCurve

# read data from csv files
train_X = pd.read_csv("training_data.csv")
test_X = pd.read_csv("test_data.csv")

# set identity column to a variable for later use
identity = test_X['IDENTITY']

train_Y = train_X['TRX_COUNT']

train_X.drop(['TRX_COUNT'], axis=1, inplace=True)

# one-hot encoding on identity values
x = pd.concat([train_X, pd.get_dummies(train_X['IDENTITY'], prefix='IDENTITY')], axis=1)
x.drop(['IDENTITY'], axis=1, inplace=True)

# adding weekend-weekday feature
x.insert(0, 'WEEKEND',
         x.apply(lambda row: 0 if (pd.Timestamp(row['YEAR'], row['MONTH'], row['DAY']).weekday() <= 5) else 1, axis=1))

# convert dataframes to an array
x_train = np.array(x)
y_train = np.array(train_Y)

# slip train data to train and test data with 0.2 ratio (80% will be train data, 20% will be test data)
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=21)

# random forest regressor as a model
model = RandomForestRegressor(n_estimators=200, max_depth=30, min_samples_split=2, min_samples_leaf=5, random_state=21)

# fit training data to the model
model.fit(X_train, Y_train)

# predict data from test data
y_predict = model.predict(X_test)

# calculate root mean squared and absolute mean errors
rmse = sqrt(mean_squared_error(Y_test, y_predict))
mae = mean_absolute_error(Y_test, y_predict)

# print errors
print("Cross validation RMSE: ", rmse)
print("Cross validation MAE: ", mae)

# one-hot encoding on identity values. Applied on test data that is read from csv file
y = pd.concat([test_X, pd.get_dummies(test_X['IDENTITY'], prefix='IDENTITY')], axis=1)
y.drop(['IDENTITY'], axis=1, inplace=True)

# adding weekend-weekday feature to test data that is read from csv file
y.insert(0, 'WEEKEND',
         y.apply(lambda row: 0 if (pd.Timestamp(row['YEAR'], row['MONTH'], row['DAY']).weekday() <= 5) else 1, axis=1))

# covert dataframe to an array
x_test = np.array(y)

# predict on test data that is read from csv file using our trained model
y_predicted = model.predict(x_test)

"""
# code snippet used for GridSearchCV

n_estimators = [10, 30, 50, 100, 200, 300 500]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

hyperF = dict(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
              min_samples_leaf=min_samples_leaf)

gridF = GridSearchCV(RandomForestRegressor(), hyperF, cv=5, verbose=1, n_jobs=-1)
bestF = gridF.fit(x_train, y_train)
print(bestF)

"""

#  give column name to the predictions
y_predicted = pd.DataFrame(y_predicted, columns=['TRX_COUNT_PRED'])

# print results to a csv files
y_predicted.to_csv('test_predictions.csv', sep='\t', encoding='utf-8', index=False)

complete_table_for_prediction = pd.concat([test_X, y_predicted], axis=1)
complete_table_for_prediction.to_csv('complete_table_for_prediction.csv', sep='\t', encoding='utf-8', index=False)

# draw graph for a specific ATM. X axis corresponds to 10 days and Y axis corresponds to number of cash withdrawals
y_predicted = np.round(y_predicted)

y_pred = pd.concat([test_X, y_predicted], axis=1)
y_pred = y_pred[(y_pred['TRX_TYPE'] == 1) & (y_pred['IDENTITY'] == 2845704036)].astype('int64').reset_index(drop=True)

x = y_pred['DAY']
y = y_pred['TRX_COUNT_PRED']

plt.plot(x, y, 'ro')
plt.xlim(0, np.max(x)+1)
plt.ylim(0, np.max(y)+20)
plt.title('Predictions for 10 days starting from where TRX_TYPE=1 / ID=2845704036')
plt.xlabel('DAYS')
plt.ylabel('CASH WITHDRAWALS')
plt.show()

print("DONE!")




