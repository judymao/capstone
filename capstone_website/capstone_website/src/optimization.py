
import numpy as np
import cvxpy as cp
from scipy.optimize import minimize


class Model:
    def __init__(self, look_ahead, goal_ret):
        self.opt_weights = 0
        self.status = None
        self.look_ahead = look_ahead
        self.goal_ret = goal_ret
        return

    def Solver(self, port, mu, Q, rf, constr_model, cost_model, risk_model, data=None, t=None, lookback=None,
               scen_model=None):

        mu_np = np.array(mu)

        Q_np = np.array(Q)

        num_stocks = port.weights.shape[1] - 1
        num_simulations = 5000

        if risk_model.risk_type == "CVAR":
            self.look_ahead = 1
            mu_np = np.array(mu)[0, :]
            mu_np = np.expand_dims(mu_np, axis=0)

        # Construct optimization problem
        all_weights = cp.Variable((num_stocks + 1, self.look_ahead))
        y = cp.Variable((num_stocks, self.look_ahead), boolean=True)
        b = cp.Variable((num_stocks, 1), boolean=True)
        z = cp.Variable((num_simulations, 1))
        g = cp.Variable(1)

        weights_prev = port.weights[-1, :-1]
        weights = all_weights[:-1, :]

        # Set model parameters
        cost_model.set_trans_cost(weights, weights_prev)
        cost_model.set_holding_cost(weights)

        if risk_model.risk_type == "CVAR":
            constr_model.set_constraints(all_weights, weights_prev, y, b, cvar=True, gamma=g, z=z, r=scen_model.value)
            risk_model.set_risk(weights, Q, S=5000, gamma=g, z=z, alpha=0.99)
        elif risk_model.risk_type == "MVO":

            constr_model.set_constraints(all_weights, weights_prev, y, b)
            risk_model.set_risk(weights, Q, self.look_ahead)

        # Get portfolio return
        portfolio_return_per_period = mu_np @ weights
        rf_return = cp.sum(rf * all_weights[-1, :])
        portfolio_return = cp.trace(portfolio_return_per_period) + rf_return

        # Max return objective
        # objective= cp.Maximize(portfolio_return-risk_model.return_adj)

        # Minimize risk objective
        objective = cp.Minimize(risk_model.value)
        constr_model.value += [
            portfolio_return - risk_model.return_adj - cost_model.trans_cost - cost_model.holding_cost >= self.goal_ret]

        # Construct Problem and Solve
        prob = cp.Problem(objective, constr_model.value)
        result = prob.solve(solver="GUROBI", verbose=False)
        self.status = prob.status

        temp_goal_ret = self.goal_ret
        counter = 0
        while self.status != "optimal":

            print("Unsolvable, Reducing Return Target")
            temp_goal_ret = 0.8 * temp_goal_ret
            if (counter > 3):
                temp_goal_ret = -0.0005

                second_counter = 0
                while self.status != "optimal":
                    temp_goal_ret = temp_goal_ret * 2
                    second_counter += 1

                    print("Temporary Goal Return is:", temp_goal_ret)
                    new_constr = [
                        portfolio_return - risk_model.return_adj - cost_model.trans_cost - cost_model.holding_cost >= temp_goal_ret]
                    constr_model.value = constr_model.value[:-1] + new_constr
                    prob = cp.Problem(objective, constr_model.value)
                    result = prob.solve(solver="GUROBI")
                    self.status = prob.status

            if self.status != "optimal":
                print("Temporary Goal Return is:", temp_goal_ret)
                new_constr = [
                    portfolio_return - risk_model.return_adj - cost_model.trans_cost - cost_model.holding_cost >= temp_goal_ret]
                constr_model.value = constr_model.value[:-1] + new_constr
                prob = cp.Problem(objective, constr_model.value)
                result = prob.solve(solver="GUROBI")
                self.status = prob.status
                counter += 1

        self.opt_weights = np.array(all_weights.value)[:, 0]
        print("Goal returns:", temp_goal_ret)
        print("port return raw:", portfolio_return.value)
        print("robustness cost:", risk_model.return_adj.value)
        print("risk value:", risk_model.value.value)
        print("holding cost:", cost_model.holding_cost.value)
        print("trans cost:", cost_model.trans_cost.value)

        return self.opt_weights

    def risk_parity(self, port, Q, lookahead, risk_model, cost_model):
        TOLERANCE = 1e-7
        Q_np = np.array(Q)
        num_stocks = port.weights.shape[1] - 1

        # Construct optimization problem
        init_weights = np.tile(port.weights[-1, :-1], lookahead).astype(float)
        init_rf = port.weights[-1, -1]
        weight_total = 1 - init_rf

        if np.count_nonzero(init_weights) == 0:
            init_weights = np.array([1 / num_stocks] * num_stocks * lookahead)
            weight_total = 1
            init_rf = 0

        # The desired contribution of each asset to the portfolio risk: we want all
        # assets to contribute equally
        assets_risk_budget = [1 / num_stocks] * num_stocks

        # Optimisation process of weights
        # Restrictions to consider in the optimisation: only long positions whose
        # sum equals 100%
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - weight_total},
                       {'type': 'ineq', 'fun': lambda x: x})

        # Optimisation process in scipy
        optimize_result = minimize(fun=risk_model.get_RP_objective,
                                   x0=init_weights,
                                   args=[Q, assets_risk_budget, lookahead, cost_model],
                                   method='SLSQP',
                                   constraints=constraints,
                                   tol=TOLERANCE,
                                   options={'disp': False, 'maxiter': 5000}
                                   )

        # Recover the weights from the optimised object
        weights = np.array(optimize_result.x)

        self.opt_weights = np.concatenate((weights[0:num_stocks], np.array([init_rf])))
        return self.opt_weights
