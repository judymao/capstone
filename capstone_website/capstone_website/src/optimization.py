
import numpy as np
import cvxpy as cp


class Model:
    def __init__(self, lam):
        self.opt_weights = 0
        self.status = None
        self.lam = lam

        return

    def MVO(self, port, mu, Q, look_ahead, constr_model, cost_model, risk_model):

        mu_np = np.array(mu)
        Q_np = np.array(Q)
        num_stocks = port.weights.shape[1]

        # Construct optimization problem
        weights = cp.Variable((num_stocks, look_ahead))
        y = cp.Variable((num_stocks, look_ahead), integer=True)

        weights_prev = port.weights[-1, :]

        # Set model parameters
        cost_model.set_trans_cost(weights, weights_prev)
        cost_model.set_holding_cost(weights)
        constr_model.set_constraints(weights, y)
        risk_model.set_risk(weights, Q, look_ahead)

        # Get portfolio return
        portfolio_return_per_period = mu_np @ weights
        portfolio_return = cp.trace(portfolio_return_per_period)

        objective = cp.Maximize(
            portfolio_return - self.lam * risk_model.value - cost_model.holding_cost - cost_model.trans_cost)

        # Construct Problem and Solve
        prob = cp.Problem(objective, constr_model.value)
        # result = prob.solve(solver="GUROBI", verbose=False)
        result = prob.solve()
        self.status = prob.status

        if self.status == "optimal":
            self.opt_weights = np.array(weights.value)[:, 1]
        else:
            self.opt_weights = weights_prev.T

        return self.opt_weights

    def BL(self):
        return
