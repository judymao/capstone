#List of imports

import pandas as pd
import scipy.optimize as sco
import numpy as np
import pandas_datareader as web
import datetime
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from hmmlearn import hmm

import statsmodels.api as smf
import urllib.request
import zipfile
import cvxpy as cp
from copy import deepcopy
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import itertools

## Additions below
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class Data:
    # Anything Data Related
    def __init__(self, stock_prices, universe, period='M'):

        if type(universe[0]) == int:
            self.stock_prices = stock_prices.iloc[:, universe]

        else:
            self.stock_prices = stock_prices[universe]

        self.stock_prices.index = pd.to_datetime(self.stock_prices.index)

        self.factor_returns = None
        self.stock_returns = self.get_stock_returns(period)

        return

    def get_stock_returns(self, period='M'):
        price = self.stock_prices.resample(period).last()

        # Calculate the percent change
        ret_data = price.pct_change()[1:]

        # Convert from series to dataframe
        ret_data = pd.DataFrame(ret_data)

        return ret_data

    def set_factor_returns(self, factor_type='FF', period='M'):
        if factor_type == 'CAPM':
            self.factor_returns = self.get_CAPM_returns(period)

        elif factor_type == 'FF':
            self.factor_returns = self.get_FF_returns(period)

        elif factor_type == 'Carhart':
            self.factor_returns = self.get_Carhart_returns(period)

        elif factor_type == 'PCA':
            self.factor_returns = self.get_PCA_returns(period)

        else:
            print("Invalid input: Please select one of the following factor types: CAPM, FF, Carhart or PCA.")

        return

    def get_FF_returns(self, period='M'):
        ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
        # Download the file and save it
        urllib.request.urlretrieve(ff_url, 'fama_french.zip')
        zip_file = zipfile.ZipFile('fama_french.zip', 'r')
        # Extact the file data
        zip_file.extractall()
        zip_file.close()
        ff_factors = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows=3, index_col=0)
        # Skip null rows
        ff_row = ff_factors.isnull().any(1).to_numpy().nonzero()[0][0]

        # Read the csv file again with skipped rows
        ff_factors = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows=3, nrows=ff_row, index_col=0)

        # Format the date index
        ff_factors.index = pd.to_datetime(ff_factors.index, format='%Y%m')

        # Format dates to end of month
        ff_factors.index = ff_factors.index + pd.offsets.MonthEnd()

        # Resample the data to correct frequency
        ff_factors = ff_factors.resample(period).last()

        # Convert from percent to decimal
        ff_factors = ff_factors.apply(lambda x: x / 100)

        return ff_factors

    def get_CAPM_returns(self, period='M'):
        ff_factors = self.get_FF_returns(period)

        # Remove the unnecessary factors
        capm_factors = ff_factors.iloc[:, 0]

        return capm_factors

    def get_Carhart_returns(self, period='M'):
        ff_factors = self.get_FF_returns(period)

        # Get the momentum factor
        momentum_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

        # Download the file and save it
        urllib.request.urlretrieve(momentum_url, 'momentum.zip')
        zip_file = zipfile.ZipFile('momentum.zip', 'r')

        # Extact the file data
        zip_file.extractall()
        zip_file.close()

        momentum_factor = pd.read_csv('F-F_Momentum_Factor.csv', skiprows=13, index_col=0)

        # Skip null rows
        row = momentum_factor.isnull().any(1).to_numpy().nonzero()[0][0]

        # Read the csv file again with skipped rows
        momentum_factor = pd.read_csv('F-F_Momentum_Factor.csv', skiprows=13, nrows=row, index_col=0)

        # Format the date index
        momentum_factor.index = pd.to_datetime(momentum_factor.index, format='%Y%m')

        # Format dates to end of month
        momentum_factor.index = momentum_factor.index + pd.offsets.MonthEnd()

        # Resample the data to correct frequency
        momentum_factor = momentum_factor.resample(period).last()

        # Convert from percent to decimal
        momentum_factor = momentum_factor.apply(lambda x: x / 100)

        # Combine to create the carhart_factors
        carhart_factors = pd.concat([ff_factors, momentum_factor], axis=1).dropna()

        return carhart_factors

    def get_PCA_returns(self, period='M'):
        exRets = self.get_stock_returns(period="D")
        num_stocks = len(exRets.columns)
        returns_mat = exRets.to_numpy()
        n_dates = returns_mat.shape[0]
        n_assets = returns_mat.shape[1]

        demeaned = (returns_mat - returns_mat.mean(axis=0)).transpose()
        sigma = 1 / (n_dates - 1) * np.matmul(demeaned, demeaned.transpose())
        eigval, eigvec = np.linalg.eig(sigma)

        principal_components = np.matmul(eigvec.transpose(), demeaned).transpose()
        pca_factors = np.real(principal_components[:, 0:100])

        pca_df = pd.DataFrame(pca_factors, index=exRets.index, columns=[str(i) for i in range(num_stocks)])
        pca_df = pca_df.resample(period).last()

        return pca_df

    def get_index_from_date(self, date_index_df, date):

        return date_index_df.index.get_loc(date)

    def get_lookback_data(self, date_index_df, date, lookback):
        end_idx = self.get_index_from_date(date_index_df, date)
        return date_index_df.iloc[end_idx - lookback:end_idx]


class Portfolio:
    # Anything Portfolio related: weights, returns, date-stamped
    def __init__(self, initial_weights):
        num_assets = len(initial_weights)
        self.weights = np.array([initial_weights])
        self.returns = np.array([])
        self.dates = []
        return

    def update_weights(self, new_weights):
        new_weights = np.array([new_weights])
        self.weights = np.append(self.weights, new_weights, axis=0)
        return

    def update_returns(self, new_returns):
        self.returns = np.append(self.returns, new_returns)
        return

    def update_dates(self, new_dates):
        self.dates.append(new_dates)
        return

    def get_Sharpe(self, risk_free):
        R_a = 0.001  # Dummy risk free rate
        sigma = np.std(self.returns - R_a)
        sharpe_ratio = (self.returns - R_a) / sigma
        return sharpe_ratio

    def plot(self):
        port_cumu_returns = np.array([x + 1 for x in self.returns]).cumprod()
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates, port_cumu_returns)
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")


class Constraints:
    # List of all constraints
    def __init__(self, constr_list, upper_limit, lower_limit, stock_limit):
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.stock_limit = stock_limit
        self.constr_list = constr_list
        self.value = []

    def get_constraints(self, weights, y):
        if "weight_unity" in self.constr_list:
            self.value += [cp.sum(weights, axis=0) == 1]

        if "cardinality" in self.constr_list:
            self.value += [cp.sum(y, axis=0) == self.stock_limit]

        if "asset_limit_cardinality" in self.constr_list:
            cardinality_upper_limit = cp.multiply(self.upper_limit, y)
            cardinality_lower_limit = cp.multiply(self.lower_limit, y)
            self.value += [weights >= cardinality_lower_limit, weights <= cardinality_upper_limit]

        if "no_short" in self.constr_list:
            self.value += [weights >= 0]

        if "asset_limit" in self.constr_list:
            self.value += [weights >= self.upper_limit, weights <= self.lower_limit]

        return


class Returns:
    def __init__(self):
        self.value = 0
        return

    def get_returns(self, mu, weights):
        portfolio_return_per_period = mu @ weights
        portfolio_return = cp.trace(portfolio_return_per_period)
        self.value = portfolio_return
        return


class Risks:
    def __init__(self, risk_type="MVO", conf_lvl=0):
        self.value = 0
        self.risk_type = risk_type
        self.conf_lvl = conf_lvl

    def get_risk(self, weights=None, Q=None, look_ahead=None, num_stocks=None):
        portfolio_risk = 0
        robustness_cost = 0
        for i in range(look_ahead):
            portfolio_risk += cp.quad_form(weights[:, i], Q[i])
        self.value = portfolio_risk

        if self.risk_type == "rect":
            for i in range(look_ahead):
                delta = stats.norm.ppf(self.conf_lvl) * np.sqrt(np.diag(Q[i] / num_stocks))
                robustness_cost += delta @ cp.abs(weights[:, i])
            self.value += robustness_cost

        elif self.risk_type == "ellip":
            for i in range(look_ahead):
                penalty = cp.norm(np.sqrt(np.diag(Q[i] / num_stocks)) @ weights[:, i], 2)

                robustness_cost += stats.chi2.ppf(self.conf_lvl, num_stocks) * penalty
            self.value += robustness_cost

        elif self.risk_type == "cvar":
            pass

        return


class Costs:
    def __init__(self, trans_coeff, holding_coeff):
        self.holding_cost = 0
        self.trans_cost = 0
        self.trans_coeff = trans_coeff
        self.holding_coeff = holding_coeff
        return

    def replicate_cost_coeff(self, num_stocks, look_ahead):
        trans_cost_repl = np.ones((num_stocks, look_ahead)) / 100
        holding_cost_repl = np.ones((num_stocks, look_ahead)) / 100
        self.trans_coeff = trans_cost_repl * self.trans_coeff
        self.holding_coeff = holding_cost_repl * self.holding_coeff
        return

    def get_holding_cost(self, weights_new):
        return cp.sum(cp.multiply(self.holding_coeff, cp.neg(weights_new)))

    def calc_trans_cost(self, weights_new, weights_old, trans_coeff):
        abs_trade = cp.abs(weights_new - weights_old)
        return cp.sum(cp.multiply(trans_coeff, abs_trade))

    def get_trans_cost(self, weights_new, weights_old):
        weights_curr = weights_new[:, 0]
        if weights_new.shape[1] > 1:
            weights_future = weights_new[:, 1:]
            weights_future_shift = weights_new[:, :-1]
            self.trans_cost = self.calc_trans_cost(weights_future, weights_future_shift, self.trans_coeff[:, 1:])
        self.trans_cost += self.calc_trans_cost(weights_curr, weights_old, self.trans_coeff[:, 0])
        return


class Model:
    def __init__(self, lam):
        self.opt_weights = 0
        self.status = None
        self.lam = lam

        return

    def MVO(self, port, mu, Q, look_ahead, constr_model, cost_model, risk_model, return_model):

        mu_np = np.array(mu)
        Q_np = np.array(Q)
        num_stocks = port.weights.shape[1]

        # Construct optimization problem
        weights = cp.Variable((num_stocks, look_ahead))
        y = cp.Variable((num_stocks, look_ahead), integer=True)

        weights_prev = port.weights[-1, :]

        cost_model.get_trans_cost(weights, weights_prev)
        cost_model.get_holding_cost(weights)

        #         constraint_model.weight_unity(weights)
        #         constraint_model.no_short(weights)
        #         constraint_model.cardinality(y)
        #         constraint_model.asset_limit_cardinality(y, weights)

        constr_model.get_constraints(weights, y)

        return_model.get_returns(mu_np, weights)

        risk_model.get_risk(weights, Q, look_ahead, num_stocks)

        # objective= cp.Maximize(return_model.value - self.lam*risk_model.value - cost_model.trans_cost - cost_model.holding_cost)
        objective = cp.Maximize(
            return_model.value - self.lam * risk_model.value - cost_model.holding_cost - cost_model.trans_cost)
        # Construct Problem and Solve
        prob = cp.Problem(objective, constr_model.value)
        result = prob.solve(solver="GUROBI")
        #         result=prob.solve()
        self.status = prob.status
        if self.status == "optimal":
            self.opt_weights = np.array(weights.value)[:, 1]
        else:
            self.opt_weights = weights_prev.T

        return self.opt_weights


class Backtest:
    def __init__(self, rebal_freq, start_date, end_date, lookback, lookahead):
        self.rebal_freq = rebal_freq
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.lookahead = lookahead
        self.reb_dates = None
        return

    def run(self, data, portfolio, optimizer, constr_model, cost, lookahead, lookback, model_type):
        stock_return = data.stock_returns
        factor_return = data.factor_returns
        self.reb_dates = np.array(data.stock_returns.loc[self.start_date:self.end_date].index)

        factor = FactorModel(data, lookahead, lookback, model_type)

        for t in self.reb_dates:
            mu, Q = factor.get_param_estimate()

            weights = optimizer.MVO(portfolio, mu, Q, constr_model, cost)

            portfolio.update_dates(t)
            portfolio.update_weights(weights)
            portfolio.update_returns(np.dot(weights, stock_return.loc[t]))

        return portfolio.get_Sharpe()

    def grid_search(self, price_data, universe, trans_coeff, hold_coeff, lam):

        #         # Overall - currently test values are used in the Backtest2 class
        #         rebalance_freqs = ['M', '3M', '6M', '12M', '60M'] # Period in Data class
        #         pot_lookaheads = [1, 3, 6, 12, 60]
        #         pot_lookbacks = [2, 3, 6, 12, 60]

        #         # Factor Models
        #         factor_models = ['CAPM', 'FF', 'Carhart', 'PCA'] # Data
        #         regressions = ['linear', 'lasso', 'ridge', 'SVR'] # FactorModel

        #         # Constraints
        #         weight_unities = ['', 'weight_unity']
        #         cardinalities = ['', 'cardinality']
        #         asset_limits = ['asset_limit_cardinality', 'asset_limit']
        #         no_shorts = ['', 'no_short']
        #         constraints_list = [weight_unities, cardinalities, asset_limits, no_shorts]

        #         stock_limits = list(range(5, 501, 5))

        #         # Optimization
        #         MVO_robustness = ['', 'rectangular', 'elliptical']

        # Overall
        rebal_freqs = ['M', '3M']
        pot_lookaheads = [5]
        pot_lookbacks = [20]

        # Factor Models
        factor_models = ['FF', 'PCA']  # Data
        regressions = ['Linear', 'Lasso', 'Ridge', 'SVR']  # FactorModel

        # Constraints
        weight_unities = ['weight_unity']
        cardinalities = ['cardinality']
        asset_limits = ['asset_limit_cardinality', 'asset_limit']
        no_shorts = ['no_short']
        constraints_list = [weight_unities, cardinalities, asset_limits, no_shorts]

        stock_limits = list(range(5, 21, 5))
        upper_asset_limits = [1]
        lower_asset_limits = [-1]

        # Optimization
        MVO_robustness = ['rectangular']

        # list of sharpe ratios per parameter combination
        sharpe_ratios = []

        # list of parameter combinations corresponding to sharpe ratio
        parameter_combos = []

        for combo in tqdm(list(itertools.product(rebal_freqs, factor_models, regressions, \
                                                 list(itertools.product(*constraints_list)), stock_limits, \
                                                 upper_asset_limits, lower_asset_limits, MVO_robustness))):

            # Store the combination
            curr_combo = {}
            curr_combo['rebalance_freq'] = combo[0]
            curr_combo['factor_model'] = combo[1]
            curr_combo['regression'] = combo[2]
            curr_combo['constraints_list'] = list(combo[3])
            curr_combo['stock_limit'] = combo[4]
            curr_combo['upper_asset_limit'] = combo[5]
            curr_combo['lower_asset_limit'] = combo[6]
            curr_combo['robustness'] = combo[7]

            # Initial Setup
            data_set = Data(price_data, universe, period=curr_combo['rebalance_freq'])
            data_set.set_factor_returns(curr_combo['factor_model'], curr_combo['rebalance_freq'])

            num_stocks = len(data_set.stock_returns.columns)
            cost_model = Costs(trans_coeff, hold_coeff)

            initial_weights = [1 / num_stocks for i in range(num_stocks)]
            port = Portfolio(initial_weights)

            # Get lookaheads that are multiples of the rebalancing frequency and <= 60 months
            if curr_combo['rebalance_freq'] == 'M':
                first = 1
            else:
                first = int(curr_combo['rebalance_freq'][0])

            lookaheads = list(itertools.compress(pot_lookaheads, [look >= first for look in pot_lookaheads]))
            lookbacks = list(itertools.compress(pot_lookbacks, [look >= first for look in pot_lookbacks]))

            for lookahead in lookaheads:
                curr_combo['lookahead'] = lookahead
                for lookback in lookbacks:
                    curr_combo['lookback'] = lookback
                    #                     print(curr_combo)

                    # Continue Setup
                    cost_model.replicate_cost_coeff(num_stocks, lookahead)
                    constr_model = Constraints(curr_combo['constraints_list'], curr_combo['upper_asset_limit'], \
                                               curr_combo['lower_asset_limit'], curr_combo['stock_limit'])

                    opt_model = Model(curr_combo['lookahead'], lam, curr_combo['robustness'])

                    # Run backtest
                    sharpe = self.run_v2(data_set, port, opt_model, constr_model, cost_model, curr_combo['lookahead'], \
                                         curr_combo['lookback'], curr_combo['regression'])

                    # Update results
                    sharpe_ratios.append(sharpe)
                    parameter_combos.append(curr_combo)

        return sharpe_ratios, parameter_combos


class FactorModel:
    def __init__(self, data, lookahead, lookback, model_type="Linear"):
        self.returns_data = data.stock_returns
        self.factor_data = data.factor_returns
        self.lookahead = lookahead
        self.lookback = lookback
        self.model_type = model_type
        return

    def get_param_estimate(self):

        if self.model_type == "LSTM":
            return self.get_mu_LSTM()

        elif self.model_type == "Linear" or self.model_type == "Lasso" or self.model_type == "Ridge" or self.model_type == "SVR":
            return self.get_mu_Q_regression()

        else:
            return "ERROR: This factor model type has not been implemented. Please input one of the following: LSTM, Linear, Lasso, Ridge, SVR."

    def get_mu_Q_regression(self):
        returns_data = self.returns_data
        factor_data = self.factor_data
        lookahead = self.lookahead
        lookback = self.lookback
        regress_type = self.model_type

        # For keeping track of mu's and Q's from each period
        mu_arr = []
        Q_arr = []

        n_factors = len(factor_data.columns)
        factor_data = factor_data.tail(lookback)
        returns_data = returns_data.tail(lookback)

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

            # Determine alpha and beta
            if regress_type == "Linear":
                model = LinearRegression().fit(X, Y)
                alpha = model.intercept_
                beta = model.coef_[:, 0:n_factors]
            elif regress_type == "Lasso":
                model = Lasso().fit(X, Y)
                alpha = model.intercept_
                beta = model.coef_[:, 0:n_factors]
            elif regress_type == "Ridge":
                model = Ridge().fit(X, Y)
                alpha = model.intercept_
                beta = model.coef_[:, 0:n_factors]
            elif regress_type == "SVR":
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
            p = 3
            sigmaEp = np.sum(epsilon ** 2, axis=0) / (len(returns_data) - 1 - p)

            #  Calculate the asset expected excess returns
            mu = model.predict([gmean])[0]

            # Calculate the diagonal matrix of residuals and the asset covariance matrix
            D = np.diag(sigmaEp)

            # Calculate the covariance matrix
            Q = np.matmul(np.matmul(beta, F.to_numpy()), beta.T) + D

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

    def get_mu_LSTM(self):
        returns_data = self.returns_data
        factor_data = self.factor_data
        lookahead = self.lookahead
        lookback = self.lookback
        regress_type = self.model_type

        tempx, tempy = self.generate_X_y(factor_data.values, returns_data.values, lookback, lookahead)
        train_x, test_x, train_y, test_y = self.traintest_split(tempx, tempy)

        # scale inputs
        scaled_train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())
        scaled_test_x = (test_x - test_x.min()) / (test_x.max() - test_x.min())
        scaled_train_y = (train_y - train_y.min()) / (train_y.max() - train_y.min())
        scaled_test_y = (test_y - test_y.min()) / (test_y.max() - test_y.min())

        mu = self.get_prediction(train_x, train_y, factor_data, lookback)
        return mu

    def generate_X_y(self, factor_data, returns_data, n_lookback, n_lookforward):
        X, y = list(), list()
        in_start = 0
        for i in range(len(factor_data)):
            in_end = in_start + n_lookback
            out_end = in_end + n_lookforward
            # ensure we have enough data for this instance
            if out_end <= len(factor_data):
                X.append(factor_data[in_start:in_end, :])
                y.append(returns_data[in_end:out_end, :])
            in_start += 1
        return np.array(X), np.array(y)

    def traintest_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def build_model(self, train_x, train_y):
        # define parameters
        verbose, epochs, batch_size = 0, 50, 16
        n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

        # define model
        model = Sequential()
        model.add(LSTM(200, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(RepeatVector(n_outputs))
        model.add(LSTM(200, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(100, activation='relu')))
        model.add(TimeDistributed(Dense(train_y.shape[2])))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)
        return model

    def forecast(self, model, history, n_lookback):
        # flatten data
        data = np.array(history)
        data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))
        # retrieve last observations for lookback data
        input_x = data[-n_lookback:, :]
        # reshape into [1, n_lookback, n]
        input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
        # forecast the next set
        yhat = model.predict(input_x, verbose=0)
        # we only want the vector forecast
        yhat = yhat[0]
        return yhat

    def evaluate_forecasts(self, actual, predicted):
        # calculate overall RMSE
        s = 0
        for row in range(actual.shape[0]):
            for col in range(actual.shape[1]):
                for k in range(actual.shape[2]):
                    s += (actual[row, col, k] - predicted[row, col, k]) ** 2
        score = sqrt(s / (actual.shape[0] * actual.shape[1] * actual.shape[2]))
        return score

    def evaluate_model(self, train_x, train_y, test_x, test_y, n_lookback):
        # fit model
        model = self.build_model(train_x, train_y)
        history = [x for x in train_x]
        # walk-forward validation
        predictions = list()
        for i in range(len(test_x)):
            yhat_sequence = self.forecast(model, history, n_lookback)
            # store the predictions
            predictions.append(yhat_sequence)
            # get real observation and add to history for predicting the next set
            history.append(test_x[i, :])
        # evaluate predictions
        predictions = np.array(predictions)
        score = self.evaluate_forecasts(test_y, predictions)
        plt.plot(model.history.history['loss'])
        # plt.plot(model.history.history['val_loss'])
        return score

    def get_prediction(self, train_x, train_y, factor_data, lookback):
        model = self.build_model(train_x, train_y)
        return self.forecast(model, factor_data.tail(lookback), lookback)

