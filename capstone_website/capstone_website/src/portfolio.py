
import numpy as np
from scipy import stats
import cvxpy as cp
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt



class Portfolio:
    # Anything Portfolio related: weights, returns, date-stamped
    def __init__(self, data):
        num_stocks = data.get_num_stocks()
        self.weights = np.array([[0] * num_stocks + [1]])  # 0 weight on stock
        self.returns = np.array([])
        self.dates = []
        return

    def update_weights(self, new_weights):
        new_weights = np.expand_dims(new_weights, axis=0)
        self.weights = np.append(self.weights, new_weights, axis=0)
        return

    def update_returns(self, new_returns):
        self.returns = np.append(self.returns, new_returns)
        return

    def update_dates(self, new_dates):
        self.dates.append(new_dates)
        return

    def get_Sharpe(self, data):
        risk_free = data.risk_free
        recent_date = self.dates[-1]
        sigma = np.std(self.returns - np.array(risk_free.loc[self.dates]))
        #         sharpe_ratio = ((np.prod(1 + self.returns - np.array(risk_free.loc[recent_date]))**(1/len(self.returns))
        #                          -1))/sigma
        sharpe_ratio = (stats.gmean(self.returns - np.array(risk_free.loc[recent_date]) + 1, axis=0) - 1) / sigma

        annual_sharpe = sharpe_ratio * (np.sqrt(12))
        return annual_sharpe



class Costs:
    def __init__(self, trans_coeff, holding_coeff):
        self.holding_cost = 0
        self.trans_cost = 0
        self.trans_coeff = trans_coeff
        self.holding_coeff = holding_coeff
        return

    def replicate_cost_coeff(self, num_stocks, lookahead):
        trans_cost_repl = np.ones((num_stocks, lookahead))
        holding_cost_repl = np.ones((num_stocks, lookahead))
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


class Constraints:
    # List of all constraints
    def __init__(self, constr_list=['asset_limit_cardinality'],
                 upper_limit=0.25, lower_limit=0, turn_over=0.25, stock_limit=11, M=0.3):
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.stock_limit = stock_limit
        self.turn_over = turn_over
        self.constr_list = constr_list
        self.M = M
        self.value = []

    def set_constraints(self, all_weights, weights_prev, y, b=None, cvar=False, gamma=None, z=None, r=None):

        # weights is without risk free
        weights = all_weights[:-1, :]

        # unity condition
        self.value += [cp.sum(all_weights, axis=0) == 1]

        # can never be short cash
        self.value += [all_weights[-1, :] >= 0]

        num_stocks = weights.shape[0]

        if cvar:
            self.value += [z >= 0]
            self.value += [z >= -r @ weights - gamma]

        if "no_short" in self.constr_list:
            self.value += [weights >= 0]

        if "turn_over" in self.constr_list:
            weight_curr = weights[:, 0]
            self.value += [cp.abs(weight_curr) - weights_prev <= self.turn_over]

        if "asset_limit_cardinality" in self.constr_list:
            upper_limit = cp.multiply(self.upper_limit, y)
            lower_limit = cp.multiply(self.lower_limit, y)

            # ensure that at least 1 but no more than 2 in each sector
            self.value += [cp.sum(y[:, 0]) <= self.stock_limit]
            self.value += [cp.sum(y[:, 0]) >= 5]
            #                 for i in range(0,num_stocks,4):
            #                     self.value += [y[i:i+3]>=1]
            # self.value += [y[i:i+3]<=2]

            #                 self.value += [weights>=lower_limit, weights<=upper_limit]
            # self.value += [weights+cp.multiply(self.M,b)>=lower_limit, -1*weights+cp.multiply(self.M,1-b)>=lower_limit]
            # self.value += [weights>= -1*upper_limit, weights<= upper_limit]
            self.value += [weights >= lower_limit, weights <= upper_limit]

        elif "asset_limit" in self.constr_list:
            self.value += [weights >= self.lower_limit, weights <= self.upper_limit]

        return

