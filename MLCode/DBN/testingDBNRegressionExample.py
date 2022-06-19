import numpy as np

np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# changed below to match where the DBN is stored on my machine
from deepBeliefNetworkMaster.dbn import SupervisedDBNRegression 


# Loading dataset
boston = load_boston()
X, Y = boston.data, boston.target

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

# Training
# BELOW IS THE WORKING REGRESSOR
# regressor = SupervisedDBNRegression(hidden_layers_structure=[100],
#                                     learning_rate_rbm=0.01,
#                                     learning_rate=0.01,
#                                     n_epochs_rbm=20,
#                                     n_iter_backprop=200,
#                                     batch_size=16,
#                                     activation_function='relu')

# BELOW IS THE BROKEN REGRESSOR (original parameters)
# regressor = SupervisedDBNRegression(hidden_layers_structure = [160, 100,30,1],learning_rate_rbm=0.05,
#                                             learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
#                                             batch_size=100, activation_function='relu')

# TESTING PARAMETERS TO FIND THE CULPRIT
# batch_size: doesn't make the difference
# n_epochs_rbm: doesn't make the difference
# learning_rate: doesn't make the difference
# learning_rate_rbm: doesn't make the difference
# hidden_layers_structure: MADE THE DIFFERENCE
regressor = SupervisedDBNRegression(hidden_layers_structure = [100],learning_rate_rbm=0.05,
                                            learning_rate=0.05, n_epochs_rbm=10, n_iter_backprop=200,
                                            batch_size=100, activation_function='relu')
regressor.fit(X_train, Y_train)

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)
print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))