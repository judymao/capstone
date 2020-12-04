

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import LSTM
# from keras.layers import RepeatVector
# from keras.layers import TimeDistributed


class FactorModel:
    def __init__(self, lookahead, lookback, regress_weighting):

        """
        lookahead: number of periods in the future to estimate
        lookback: number of periods in the past to use for estimations
        regress_weighting: array of size 4 with weight corresponding to each regression type; adds up to 1;
        order is linear, lasso, ridge, SVR; in the case where there is one 1 and the rest 0's, there is no ensembling;
        can artifically call LSTM by setting all weights to 0
        """
        self.lookahead = lookahead
        self.lookback = lookback
        self.regress_weighting = regress_weighting
        return

    def get_param_estimate(self, rebal_date, data):

        if sum(self.regress_weighting) == 0:
            return self.get_mu_LSTM(rebal_date, data)

        elif sum(self.regress_weighting) == 1:
            return self.get_mu_Q_regression(rebal_date, data)

        else:
            return "ERROR: This regression weighting is not valid. Please make sure the weights sum to 1. You can also give all zeros for LSTM."

    def get_mu_Q_regression(self, rebal_date, data):
        returns_data = data.stock_returns
        factor_data = data.factor_returns
        lookahead = self.lookahead
        lookback = self.lookback
        regress_weighting = self.regress_weighting

        # For keeping track of mu's and Q's from each period
        mu_arr = []
        Q_arr = []

        n_factors = len(factor_data.columns)

        returns_data = data.get_lookback_data(returns_data, rebal_date, lookback)
        factor_data = data.get_lookback_data(factor_data, rebal_date, lookback)

        for i in range(0, lookahead):
            # Calculate the factor covariance matrix

            F = factor_data.loc[:, factor_data.columns != 'Ones'].cov()

            # Calculate the factor expected excess return from historical data using the geometric mean
            factor_data['Ones'] = [1 for i in range(len(factor_data))]
            gmean = stats.gmean(factor_data + 1, axis=0) - 1

            # Set up X and Y to determine alpha and beta
            X = factor_data
            Y = returns_data
            X = X.to_numpy()
            Y = Y.to_numpy()

            ### LINEAR REGRESSION

            model = LinearRegression().fit(X, Y)
            alpha = model.intercept_
            beta = model.coef_[:, 0:n_factors]

            # Calculate the residuals
            alpha = np.reshape(alpha, (alpha.size, 1))
            epsilon = returns_data.to_numpy() - np.matmul(X, np.transpose(np.hstack((beta, alpha))))

            # Calculate the residual variance with "N - p - 1" degrees of freedom
            sigmaEp = np.sum(epsilon ** 2, axis=0) / (len(returns_data) - n_factors - 1)

            #  Calculate the asset expected excess returns
            mu_linear = model.predict([gmean])[0]

            # Calculate the diagonal matrix of residuals and the asset covariance matrix
            D = np.diag(sigmaEp)

            # Calculate the covariance matrix
            Q_linear = np.matmul(np.matmul(beta, F.to_numpy()), beta.T) + D

            ### LASSO REGRESSION

            model = Lasso().fit(X, Y)
            alpha = model.intercept_
            beta = model.coef_[:, 0:n_factors]

            # Calculate the residuals
            alpha = np.reshape(alpha, (alpha.size, 1))
            epsilon = returns_data.to_numpy() - np.matmul(X, np.transpose(np.hstack((beta, alpha))))

            # Calculate the residual variance with "N - p - 1" degrees of freedom
            sigmaEp = np.sum(epsilon ** 2, axis=0) / (len(returns_data) - n_factors - 1)

            #  Calculate the asset expected excess returns
            mu_lasso = model.predict([gmean])[0]

            # Calculate the diagonal matrix of residuals and the asset covariance matrix
            D = np.diag(sigmaEp)

            # Calculate the covariance matrix
            Q_lasso = np.matmul(np.matmul(beta, F.to_numpy()), beta.T) + D

            ### RIDGE REGRESSION

            model = Ridge().fit(X, Y)
            alpha = model.intercept_
            beta = model.coef_[:, 0:n_factors]

            # Calculate the residuals
            alpha = np.reshape(alpha, (alpha.size, 1))
            epsilon = returns_data.to_numpy() - np.matmul(X, np.transpose(np.hstack((beta, alpha))))

            # Calculate the residual variance with "N - p - 1" degrees of freedom
            sigmaEp = np.sum(epsilon ** 2, axis=0) / (len(returns_data) - n_factors - 1)

            #  Calculate the asset expected excess returns
            mu_ridge = model.predict([gmean])[0]

            # Calculate the diagonal matrix of residuals and the asset covariance matrix
            D = np.diag(sigmaEp)

            # Calculate the covariance matrix
            Q_ridge = np.matmul(np.matmul(beta, F.to_numpy()), beta.T) + D

            ### SUPPORT VECTOR REGRESSION

            model = make_pipeline(StandardScaler(), MultiOutputRegressor(
                LinearSVR(C=1, dual=False, loss="squared_epsilon_insensitive"))).fit(X, Y)
            beta = np.array([[model.named_steps['multioutputregressor'].estimators_[i].coef_[0:n_factors] for i in
                              range(len(model.named_steps['multioutputregressor'].estimators_))]])[0]
            alpha = np.array([model.named_steps['multioutputregressor'].estimators_[i].intercept_[0] for i in
                              range(len(model.named_steps['multioutputregressor'].estimators_))])

            # Calculate the residuals
            alpha = np.reshape(alpha, (alpha.size, 1))
            epsilon = returns_data.to_numpy() - np.matmul(X, np.transpose(np.hstack((beta, alpha))))

            # Calculate the residual variance with "N - p - 1" degrees of freedom
            sigmaEp = np.sum(epsilon ** 2, axis=0) / (len(returns_data) - n_factors - 1)

            #  Calculate the asset expected excess returns
            mu_SVR = model.predict([gmean])[0]

            # Calculate the diagonal matrix of residuals and the asset covariance matrix
            D = np.diag(sigmaEp)

            # Calculate the covariance matrix
            Q_SVR = np.matmul(np.matmul(beta, F.to_numpy()), beta.T) + D

            # Ensemble the methods
            mu = regress_weighting[0] * mu_linear + regress_weighting[1] * mu_lasso + regress_weighting[2] * mu_ridge + \
                 regress_weighting[3] * mu_SVR
            Q = regress_weighting[0] * Q_linear + regress_weighting[1] * Q_lasso + regress_weighting[2] * Q_ridge + \
                regress_weighting[3] * Q_SVR

            # Add mu and Q to array
            mu_arr.append(mu)
            Q_arr.append(Q)

            # Update for next time step
            factor_data = factor_data[1:]
            factor_append = pd.Series(gmean, index=factor_data.columns)
            factor_data = factor_data.append(factor_append, ignore_index=True)

            returns_data = returns_data[1:]
            mu_append = pd.Series(mu, index=returns_data.columns)
            returns_data = returns_data.append(mu_append, ignore_index=True)

        return mu_arr, Q_arr

    # def get_mu_LSTM(self, rebal_date, data):
    #     returns_data = data.stock_returns
    #     factor_data = data.factor_returns
    #
    #     lookahead = self.lookahead
    #     lookback = self.lookback
    #     regress_weighting = self.regress_weighting
    #
    #     returns_data = data.get_lookback_data(returns_data, rebal_date, lookback)
    #     factor_data = data.get_lookback_data(factor_data, rebal_date, lookback)
    #
    #     tempx, tempy = self.generate_X_y(factor_data.values, returns_data.values, lookback, lookahead)
    #     train_x, test_x, train_y, test_y = self.traintest_split(tempx, tempy)
    #
    #     # scale inputs
    #     scaled_train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
    #     scaled_test_x = (test_x - test_x.min()) / (test_x.max() - test_x.min())
    #     scaled_train_y = (train_y - train_y.min()) / (train_y.max() - train_y.min())
    #     scaled_test_y = (test_y - test_y.min()) / (test_y.max() - test_y.min())
    #
    #     mu = self.get_prediction(train_x, train_y, factor_data, lookback)
    #     return mu
    #
    # def generate_X_y(self, factor_data, returns_data, n_lookback, n_lookforward):
    #     X, y = list(), list()
    #     in_start = 0
    #     for i in range(len(factor_data)):
    #         in_end = in_start + n_lookback
    #         out_end = in_end + n_lookforward
    #         # ensure we have enough data for this instance
    #         if out_end <= len(factor_data):
    #             X.append(factor_data[in_start:in_end, :])
    #             y.append(returns_data[in_end:out_end, :])
    #         in_start += 1
    #     return np.array(X), np.array(y)
    #
    # def traintest_split(self, X, y):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    #     return X_train, X_test, y_train, y_test
    #
    # def build_model(self, train_x, train_y):
    #     # define parameters
    #     verbose, epochs, batch_size = 0, 50, 16
    #     n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
    #
    #     # define model
    #     model = Sequential()
    #     model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
    #     model.add(RepeatVector(n_outputs))
    #     model.add(LSTM(200, activation='relu', return_sequences=True))
    #     model.add(TimeDistributed(Dense(100, activation='relu')))
    #     model.add(TimeDistributed(Dense(train_y.shape[2])))
    #     model.compile(loss='mse', optimizer='adam')
    #     # fit network
    #     model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
    #     return model
    #
    # def forecast(self, model, history, n_lookback):
    #     # flatten data
    #     data = np.array(history)
    #     data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
    #     # retrieve last observations for lookback data
    #     input_x = data[-n_lookback:, :]
    #     # reshape into [1, n_lookback, n]
    #     input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
    #     # forecast the next set
    #     yhat = model.predict(input_x, verbose=0)
    #     # we only want the vector forecast
    #     yhat = yhat[0]
    #     return yhat
    #
    # def evaluate_forecasts(self, actual, predicted):
    #     # calculate overall RMSE
    #     s = 0
    #     for row in range(actual.shape[0]):
    #         for col in range(actual.shape[1]):
    #             for k in range(actual.shape[2]):
    #                 s += (actual[row, col, k] - predicted[row, col, k]) ** 2
    #     score = sqrt(s / (actual.shape[0] * actual.shape[1] * actual.shape[2]))
    #     return score
    #
    # def evaluate_model(self, train_x, train_y, test_x, test_y, n_lookback):
    #     # fit model
    #     model = self.build_model(train_x, train_y)
    #     history = [x for x in train_x]
    #     # walk-forward validation
    #     predictions = list()
    #     for i in range(len(test_x)):
    #         yhat_sequence = self.forecast(model, history, n_lookback)
    #         # store the predictions
    #         predictions.append(yhat_sequence)
    #         # get real observation and add to history for predicting the next set
    #         history.append(test_x[i, :])
    #     # evaluate predictions
    #     predictions = np.array(predictions)
    #     score = self.evaluate_forecasts(test_y, predictions)
    #     plt.plot(model.history.history['loss'])
    #     # plt.plot(model.history.history['val_loss'])
    #     return score
    #
    # def get_prediction(self, train_x, train_y, factor_data, lookback):
    #     model = self.build_model(train_x, train_y)
    #     return self.forecast(model, factor_data.tail(lookback), lookback)
