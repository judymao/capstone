'''
For generating the three baseline portfolios
'''


from capstone_website.models import Stock, PortfolioInfo, PortfolioData
from capstone_website import db



def generate_baseline_portfolios():

    print(f"Generating the baseline portfolios!")

    initial_cash = 100
    user_id = 1
    time_horizon = 12
    portfolio_id = 10000  # Set really large portfolio id to not overlap with existing entries

    risk_mapping = {0: "Baseline_Low",
                    1: "Baseline_Medium",
                    2: "Baseline_High"}

    for risk_level in risk_mapping.keys():

        print(f"Portfolio_id: {portfolio_id}")
        portfolio_name = risk_mapping[risk_level]
        risk_appetite = portfolio_name.split("_")[1]
        baseline_portfolio = PortfolioInfo.get_baseline_portfolios(risk_appetite)

        if baseline_portfolio is None:
            is_error = True
            while is_error:
                try:
                    portfolio_info = PortfolioInfo(id=portfolio_id, user_id=user_id, win_philosophy=risk_level,
                                                   lose_philosophy=risk_level, games_philosophy=risk_level,
                                                   unknown_philosophy=risk_level, job_philosophy=risk_level,
                                                   monitor_philosophy=risk_level, name=portfolio_name,
                                                   time_horizon=time_horizon, cash=initial_cash)
                    db.session.add(portfolio_info)
                    db.session.commit()
                    is_error = False
                except Exception as e:
                    portfolio_id += risk_level
                    print(f"Failed to store portfolio info with exception: {e}, trying a new portfolio_id: {portfolio_id}")

            portfolio_data_list = portfolio_info.create_portfolio()
            db.session.add_all(portfolio_data_list)
            db.session.commit()
            print(f"Created a portfolio with risk level: {risk_appetite} with {len(portfolio_data_list)} entries")

        elif baseline_portfolio.time_horizon != float(time_horizon):
            print(f"Baseline exists with time horizon: {baseline_portfolio.time_horizon}, updating to time horizon: {time_horizon}")
            try:
                PortfolioInfo.query.filter_by(name="Baseline_" + risk_appetite).update({"time_horizon": time_horizon})
                portfolio_info.time_horizon = time_horizon
            except Exception as e:
                print(f"Failed to update portfolio info with exception: {e}")

            portfolio_data_list = portfolio_info.create_portfolio()
            db.session.add_all(portfolio_data_list)
            db.session.commit()
            print(f"Created a portfolio with risk level: {risk_appetite} with {len(portfolio_data_list)} entries")

        else:
            print(f"Already detected a portfolio with risk level: {risk_appetite} | Time horizon: {baseline_portfolio.time_horizon:,.0f} years, Returns: {baseline_portfolio.returns:,.2%} and volatility: {baseline_portfolio.volatility:,.2%}")

        portfolio_id += risk_level + 1


if __name__ == "__main__":
    generate_baseline_portfolios()