
import numpy as np
from scipy import stats
import cvxpy as cp
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt


class Portfolio:
    # Anything Portfolio related: weights, returns, date-stamped
    def __init__(self, data):
        num_stocks = data.get_num_stocks()
        self.weights = np.array([[0] * (num_stocks - 1) + [1]])  # 0 weight on stock
        #         self.rf_weight = np.array([1]) # 1 in risk-free market
        self.returns = np.array([])
        self.dates = []
        #         self.rf_rates = data.risk_free
        return

    def update_weights(self, new_weights):
        new_weights = np.array([new_weights])
        self.weights = np.append(self.weights, new_weights, axis=0)
        #         self.rf_weights = np.append(rf_weights)
        return

    def update_returns(self, new_returns):
        self.returns = np.append(self.returns, new_returns)
        return

    def update_dates(self, new_dates):
        self.dates.append(new_dates)
        return

    def get_Sharpe(self):
        recent_date = self.dates[-1]
        #         sigma = np.std(self.returns - self.risk_free[self.dates])
        #         sharpe_ratio = ((np.prod(1+self.returns)-1) - self.risk_free[recent_date])/sigma
        sigma = np.std(self.returns - 0.01)
        sharpe_ratio = ((np.prod(1 + self.returns) - 1) - 0.01) / sigma
        return sharpe_ratio

    def plot(self):
        port_cumu_returns = np.array([x + 1 for x in self.returns]).cumprod()
        plt.figure(figsize=(12, 6))
        plt.plot(self.dates, port_cumu_returns)
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.show()


class Constraints:
    # List of all constraints
    def __init__(self, constr_list=['cardinality', 'asset_limit_cardinality', 'no_short'],
                 upper_limit=1, lower_limit=-1, stock_limit=20):
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.stock_limit = stock_limit
        self.constr_list = constr_list
        self.value = []

    def set_constraints(self, weights, y):
        # Set weight unity
        self.value += [cp.sum(weights, axis=0) == 1]

        if "cardinality" in self.constr_list:
            self.value += [cp.sum(y, axis=0) == self.stock_limit]

        if "no_short" in self.constr_list:
            self.value += [weights >= 0]

        if "asset_limit_cardinality" in self.constr_list:
            cardinality_upper_limit = cp.multiply(self.upper_limit, y)
            cardinality_lower_limit = cp.multiply(self.lower_limit, y)
            self.value += [weights >= cardinality_lower_limit, weights <= cardinality_upper_limit]

        elif "asset_limit" in self.constr_list:
            self.value += [weights >= self.upper_limit, weights <= self.lower_limit]

        return


class Risks:
    def __init__(self, risk_type="MVO", conf_lvl=0):
        self.value = 0
        self.risk_type = risk_type
        self.conf_lvl = conf_lvl
        return

    def set_risk(self, weights, Q, lookahead):
        portfolio_risk = 0
        robustness_cost = 0
        num_stocks = weights.shape[1]

        for i in range(lookahead):
            portfolio_risk += cp.quad_form(weights[:, i], Q[i])
        self.value = portfolio_risk

        if self.risk_type == "rect":
            for i in range(lookahead):
                delta = stats.norm.ppf(self.conf_lvl) * np.sqrt(np.diag(Q[i] / num_stocks))
                robustness_cost += delta @ cp.abs(weights[:, i])
            self.value += robustness_cost

        elif self.risk_type == "ellip":
            for i in range(lookahead):
                penalty = cp.norm(np.sqrt(np.diag(Q[i] / num_stocks)) @ weights[:, i], 2)

                robustness_cost += stats.chi2.ppf(self.conf_lvl, num_stocks) * penalty
            self.value += robustness_cost

        elif self.risk_type == "cvar":
            pass

        elif self.risk_type == 'B-L':
            self.value = 0
            pass

        return


class Costs:
    def __init__(self, trans_coeff, holding_coeff):
        self.holding_cost = 0
        self.trans_cost = 0
        self.trans_coeff = trans_coeff
        self.holding_coeff = holding_coeff
        return

    def replicate_cost_coeff(self, num_stocks, lookahead):
        trans_cost_repl = np.ones((num_stocks, lookahead)) / 100
        holding_cost_repl = np.ones((num_stocks, lookahead)) / 100
        self.trans_coeff = trans_cost_repl * self.trans_coeff
        self.holding_coeff = holding_cost_repl * self.holding_coeff
        return

    def set_holding_cost(self, weights_new):
        self.holding_cost += cp.sum(cp.multiply(self.holding_coeff, cp.neg(weights_new)))
        return

    def calc_trans_cost(self, weights_new, weights_old, trans_coeff):
        abs_trade = cp.abs(weights_new - weights_old)
        return cp.sum(cp.multiply(trans_coeff, abs_trade))

    def set_trans_cost(self, weights_new, weights_old):
        weights_curr = weights_new[:, 0]
        if weights_new.shape[1] > 1:
            weights_future = weights_new[:, 1:]
            weights_future_shift = weights_new[:, :-1]
            self.trans_cost = self.calc_trans_cost(weights_future, weights_future_shift, self.trans_coeff[:, 1:])

        self.trans_cost += self.calc_trans_cost(weights_curr, weights_old, self.trans_coeff[:, 0])
        return

