

import numpy as np
from scipy import stats
from scipy.stats import skew, pearson3
from sklearn.linear_model import Ridge
import cvxpy as cp
import random

random.seed(50)

## Additions below
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler



class Risks:
    def __init__(self, risk_type="MVO", robust_type="ellip", conf_lvl=0):
        # risk value, return adjustment, risk type and confidence level
        self.value = 0
        self.return_adj = 0
        self.risk_type = risk_type
        self.robust_type = robust_type
        self.conf_lvl = conf_lvl
        return

    def set_risk(self, weights, Q, lookahead=1, S=5000, gamma=None, z=None, alpha=None):

        portfolio_risk = 0
        robustness_cost = 0
        num_stocks = weights.shape[1]

        if self.risk_type == "MVO":

            for i in range(lookahead):
                portfolio_risk += cp.quad_form(weights[:, i], Q[i])
            self.value = portfolio_risk

        elif self.risk_type == "CVAR":
            if not S or not gamma or not z or not alpha:
                print("Missing one of these required inputs for CVaR optimization: S, gamma, z, alpha")
                return
            self.value = gamma + (1 / ((1 - alpha) * S)) * cp.sum(z)

        if self.robust_type == "rect":

            for i in range(lookahead):
                delta = stats.norm.ppf(self.conf_lvl) * np.sqrt(np.diag(Q[i] / num_stocks))
                robustness_cost += delta @ cp.abs(weights[:, i])

            self.return_adj = robustness_cost

        elif self.robust_type == "ellip":

            for i in range(lookahead):
                penalty = cp.norm(np.sqrt(np.diag(Q[i] / num_stocks)) @ weights[:, i], 2)
                robustness_cost += stats.chi2.ppf(self.conf_lvl, num_stocks) * penalty

            self.return_adj = robustness_cost

        return

    def get_RP_objective(self, weights, args):
        Q = args[0]
        assets_risk_budget = args[1]
        lookahead = args[2]
        cost_model = args[3]

        num_stocks = len(assets_risk_budget)

        self.value = 0
        # We convert the weights to a matrix
        weights = np.matrix(weights)
        for i in range(lookahead):
            # We calculate the risk of the weights distribution

            portfolio_risk = np.sqrt((weights[0, num_stocks * i:num_stocks * (i + 1)] * Q[i]
                                      * weights[0, num_stocks * i:num_stocks * (i + 1)].T))[0, 0]

            # We calculate the contribution of each asset to the risk of the weights
            # distribution
            assets_risk_contribution = np.multiply(weights[0, num_stocks * i:num_stocks * (i + 1)].T, Q[i]
                                                   * weights[0, num_stocks * i:num_stocks * (i + 1)].T) / portfolio_risk

            # We calculate the desired contribution of each asset to the risk of the
            # weights distribution
            assets_risk_target = np.asmatrix(np.multiply(portfolio_risk, assets_risk_budget))

            # Error between the desired contribution and the calculated contribution of
            # each asset
            self.value += np.sum(np.square(assets_risk_contribution - assets_risk_target.T))

            # Get the holding costs
            self.value += np.sum(cost_model.holding_coeff[0, 0] * weights[0, num_stocks * i:num_stocks * (i + 1)])

            # Get the transaction costs
            if i < lookahead - 1:
                abs_trade = np.abs(weights[0, num_stocks * i:num_stocks * (i + 1)] -
                                   weights[0, num_stocks * (i + 1):num_stocks * (i + 2)])
                self.value += np.sum(cost_model.trans_coeff[0, 0] * abs_trade)

        # It returns the calculated error
        return self.value

class Scenarios:
    def __init__(self, mode):
        self.mode=mode
        self.value=None

    def gen_scenarios(self, S, data, t, lookback):
        if self.mode==0:
            factor_returns = data.get_factor_returns()
            prev_factor_returns = factor_returns[:t]

            mu_simulated_arr = []

            returns_data = data.stock_returns
            factor_data = data.factor_returns

            n_factors = len(factor_data.columns)

            returns_data = data.get_lookback_data(returns_data, t, lookback)
            factor_data = data.get_lookback_data(factor_data, t, lookback)

            factor_data['Ones'] = [1 for i in range(len(factor_data))]


            # Set up X and Y to determine alpha and beta
            X = factor_data
            Y = returns_data
            X = X.to_numpy()
            Y = Y.to_numpy()


            # RIDGE REGRESSION
            model_ridge = Ridge().fit(X,Y)

            # SUPPORT VECTOR REGRESSION
            model_SVR = make_pipeline(StandardScaler(), MultiOutputRegressor(LinearSVR(C=1, dual=False, loss="squared_epsilon_insensitive"))).fit(X, Y)


            sim_list = []

            for j in range(n_factors):
                fac_ret = prev_factor_returns.iloc[:,j].to_numpy()
                skew_val = skew(fac_ret)
                mean_val = np.mean(fac_ret)
                std_val = np.std(fac_ret)
                sim_list.append(pearson3.rvs(skew=skew_val, loc=mean_val, scale=std_val, size=S))

            sim_list.append(np.ones(S))

            mu_simulated_arr = []
            for i in range(S):
                #  Calculate the asset expected excess returns
                mu_ridge = model_ridge.predict([np.array(sim_list)[:,i]])[0]
                mu_SVR = model_SVR.predict([np.array(sim_list)[:,i]])[0]

                # Ensemble the methods
                mu = 0.25*mu_ridge + 0.75*mu_SVR

                mu_simulated_arr.append(mu)

            self.value= np.array(mu_simulated_arr)

        elif self.mode==1:
            returns_data = data.stock_returns
            rets = data.get_lookback_data(returns_data, t, lookback)
            num_stocks=data.get_num_stocks()
            num_dates=len(rets.index)
            num_scen=S
            dof=4
            mat_idx= np.random.randint(0,num_dates-1,(num_scen,30))
            chi_squared_mult= np.sqrt(dof/np.random.chisquare(dof,(num_scen,1)))
            rets_rank=rets.rank()/100
            rets_gauss= stats.norm.ppf(rets_rank)
            mat_gauss=rets_gauss[mat_idx]
            gauss_sum=mat_gauss.sum(axis=1)
            t_student=np.multiply(gauss_sum,chi_squared_mult)
            copula= stats.t.cdf(t_student,dof)
            empir=np.zeros((num_scen, num_stocks))
            for i in range(num_stocks):
                empir[:,i]=np.quantile(rets.iloc[:,i].T, copula[:,i])
            self.value=empir