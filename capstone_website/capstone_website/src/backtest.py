
import numpy as np
import itertools

from capstone_website.src.factor_models import FactorModel
from capstone_website.src.portfolio import Costs, Constraints, Risks


class Backtest:
    def __init__(self, start_date, end_date, lookback, lookahead):
        self.rebal_freq = 'M'
        self.start_date = start_date
        self.end_date = end_date
        self.lookback = lookback
        self.lookahead = lookahead
        self.reb_dates = None
        return

    def run(self, data, portfolio, factor_model, optimizer, constr_model, cost_model, risk_model):
        stock_return = data.stock_returns
        self.reb_dates = np.array(data.stock_returns.loc[self.start_date:self.end_date].index)

        for t in self.reb_dates:
            mu, Q = factor_model.get_param_estimate(t, data)

            if risk_model.risk_type in ('MVO', 'ellip', 'rect', 'cvar'):
                weights = optimizer.MVO(portfolio, mu, Q, self.lookahead, constr_model, cost_model, risk_model)

            elif risk_model.risk_type == 'B-L':
                weights = optimizer.BL()

            portfolio.update_dates(t)
            portfolio.update_weights(weights)
            portfolio.update_returns(np.dot(weights, stock_return.loc[t]))

            ##How the lambdas influence
            ##When to use MVO.. when to use CVaR... implement a CVaR

        return portfolio.get_Sharpe()

    def grid_search(self, data, portfolio, model, trans_coeff=0.2, hold_coeff=0.2, lam=0.9, conf_level=0.95):

        # Overall
        pot_lookaheads = [5]
        pot_lookbacks = [20]

        # Factor Models
        factor_models = ['FF', 'PCA']  # Data
        weights = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0.25, 0.25, 0.25, 0.25],
                   [0, 0.5, 0.5, 0], [0, 0.25, 0.25, 0.5], [0, 0, 0.5, 0.5]]

        # Constraints
        cardinalities = ['cardinality']
        asset_limits = ['asset_limit_cardinality', 'asset_limit']
        no_shorts = ['no_short']
        constraints_list = [cardinalities, asset_limits, no_shorts]

        stock_limits = list(range(5, 21, 5))
        upper_asset_limits = [1]
        lower_asset_limits = [-1]

        # Optimization
        MVO_robustness = ['ellip']

        # list of sharpe ratios per parameter combination
        sharpe_ratios = []

        # list of parameter combinations corresponding to sharpe ratio
        parameter_combos = []

        for combo in (list(itertools.product(factor_models, weights, list(itertools.product(*constraints_list)),
                                             stock_limits, upper_asset_limits, lower_asset_limits, MVO_robustness))):

            # Store the combination
            curr_combo = {'rebalance_freq': 'M', 'factor_model': combo[0], 'weights': combo[1],
                          'constraints_list': list(combo[2]), 'stock_limit': combo[3], 'upper_asset_limit': combo[4],
                          'lower_asset_limit': combo[5], 'robustness': combo[6]}

            # Initial Setup
            data.set_factor_returns(curr_combo['factor_model'], curr_combo['rebalance_freq'])

            num_stocks = data.get_num_stocks()
            cost_model = Costs(trans_coeff, hold_coeff)

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

                    # Continue Setup
                    cost_model.replicate_cost_coeff(num_stocks, lookahead)
                    constr_model = Constraints(curr_combo['constraints_list'])

                    risk_model = Risks(curr_combo['robustness'], conf_level)

                    # Run backtest
                    factor = FactorModel(curr_combo['lookahead'], curr_combo['lookback'],
                                         curr_combo['weights'])
                    sharpe = self.run(data, portfolio, factor, model, constr_model, cost_model, risk_model)

                    # Update results
                    sharpe_ratios.append(sharpe)
                    parameter_combos.append(curr_combo)

        return sharpe_ratios, parameter_combos
