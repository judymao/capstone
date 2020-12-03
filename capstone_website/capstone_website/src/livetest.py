

import numpy as np
from scipy import stats
from hmmlearn import hmm
import random

# from capstone_website.src.factor_models import
from capstone_website.src.portfolio import Costs, Constraints
from capstone_website.src.risk_models import Risks, Scenarios
from capstone_website.src.optimization import Model


random.seed(50)



class Regime:
    def __init__(self, data, t, first_date="2000-01-01"):
        self.train_prices = None
        self.train_returns = None
        self.train_dates = None
        self.get_train_data(data, t, first_date)

    def get_train_data(self, data, t, first_date):
        mkt_data = data.factor_returns["Mkt-RF"] + data.factor_returns["RF"]
        mkt_returns = mkt_data[first_date:t]
        self.train_dates = mkt_returns.index

        mkt_returns = np.array(mkt_returns.values)
        mkt_prices = 100 * (np.array([x + 1 for x in mkt_returns]).cumprod())
        mkt_prices = np.expand_dims(mkt_prices, axis=1)
        mkt_returns = np.expand_dims(mkt_returns, axis=1)

        self.train_prices = mkt_prices
        self.train_returns = mkt_returns

    def HMM(self, num_hs):
        model = hmm.GaussianHMM(n_components=num_hs)
        model.fit(self.train_returns)
        return model

    def predict_next(self, num_hs=2):
        reg_model = self.HMM(num_hs)
        out = reg_model.predict(self.train_returns)
        transmat = reg_model.transmat_
        gauss_means = reg_model.means_
        gauss_cov = reg_model.covars_

        bull_idx = np.argmax(reg_model.means_)
        bear_idx = 1 - bull_idx
        seq = reg_model.predict(self.train_returns)
        next_state = np.argmax(transmat[seq[-1]])
        if bull_idx == next_state:
            self.reg_pred = "bull"

        else:
            self.reg_pred = "bear"

        self.trans_conf = transmat[seq[-1]][next_state]
        self.reg_conf = \
        stats.norm.cdf(self.train_returns[-1], gauss_means[next_state], np.sqrt(gauss_cov[next_state]))[0][0]


class Livetest:
    def __init__(self, start_date, end_date, period='M'):
        self.rebal_freq = period
        self.start_date = start_date
        self.end_date = end_date

        return

    def run(self, data, FF_data, portfolio, risk_profile, factor_model, u_lim=0.25, l_lim=0.05, stock_lim=11,
            constraints=['asset_limit_cardinality']):
        look_back = factor_model.lookback
        look_ahead = factor_model.lookahead
        stock_return = data.stock_returns
        num_stocks = data.get_num_stocks()
        reb_dates = np.array(data.stock_returns.loc[self.start_date:self.end_date].index)

        for t in reb_dates:
            # need some dynamic adjustment here... how to set risk_model confidence level, cost_coefficieints,
            # constraint asset limits, goal_return ,, factor_model, opt_model, constr_model, cost_model, risk_model

            # detect regime
            hmm_model = Regime(FF_data, t)
            hmm_model.predict_next()

            # ADJUST THIS STUFF BELOW
            if hmm_model.reg_pred == "bull":
                # CVaR here
                # regime dependent coefficients
                # print("trans_conf:", hmm_model.trans_conf)
                # print("reg_conf", hmm_model.reg_conf)
                regime_ret_adj = 1 + 0.1 * hmm_model.trans_conf + 0.1 * hmm_model.reg_conf
                raw_return = (1 + risk_profile) ** (1 / 12) - 1
                adj_return = regime_ret_adj * raw_return
                trans_coeff = 0.01
                holding_coeff = 0.05
                conf_level = 0.25
                #                 constr_list = ["asset_limit_cardinality"]
                constr_list = constraints
                risk_model = Risks("MVO", "ellip", conf_level)
                scen_model = None



            elif hmm_model.reg_pred == "bear":
                # MVO here
                # regime dependent coefficients
                regime_ret_adj = 1 - 0.1 * hmm_model.trans_conf - 0.1 * hmm_model.reg_conf
                raw_return = (1 + risk_profile) ** (1 / 12) - 1
                adj_return = regime_ret_adj * raw_return
                trans_coeff = 0.01
                holding_coeff = 0.01
                conf_level = 0.75
                #                 constr_list = ["asset_limit_cardinality"]
                constr_list = constraints
                risk_model = Risks("CVAR", "ellip", conf_level)
                scen_model = Scenarios(1)
                scen_model.gen_scenarios(5000, data, t, look_back)

            # Set up models based on regime hyperparameters
            constr_model = Constraints(constr_list, upper_limit=u_lim,
                                       lower_limit=l_lim, stock_limit=stock_lim)

            # Set up cost models
            cost_model = Costs(trans_coeff, holding_coeff)
            cost_model.replicate_cost_coeff(num_stocks, look_ahead)

            opt_model = Model(look_ahead, raw_return)
            # risk_model = Risks("CVAR", "ellip", conf_level)
            # scen_model = Scenarios(1)
            # scen_model.gen_scenarios(5000, data, t, look_back)

            mu, Q = factor_model.get_param_estimate(t, data)
            new_rf_rate = float(data.risk_free.loc[t])

            weights = opt_model.Solver(portfolio, mu, Q, new_rf_rate, constr_model, cost_model, risk_model, data, t,
                                       look_back, scen_model)

            portfolio.update_dates(t)
            portfolio.update_weights(weights)
            portfolio.update_returns(np.dot(weights[:-1], stock_return.loc[t]) + weights[-1] * new_rf_rate)

        return portfolio.get_Sharpe(data)
