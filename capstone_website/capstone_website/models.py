from capstone_website import db, login_manager, client
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import tiingo
from datetime import date, timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np

from capstone_website.src.constants import Constants
from capstone_website.src.data import Data
from capstone_website.src.portfolio import Portfolio, Constraints, Costs, Risks
from capstone_website.src.optimization import Model
from capstone_website.src.factor_models import FactorModel
from capstone_website.src.backtest import Backtest


constants = Constants()

class User(UserMixin, db.Model):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(16), unique=True)
    email = db.Column(db.String(255), unique=True)
    first_name = db.Column(db.String(255))
    last_name = db.Column(db.String(255))
    company = db.Column(db.String(255))
    password_hash = db.Column(db.String(128))

    def __repr__(self):
        return '<User %r>' % self.user

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)


class Stock(db.Model):
    __tablename__ = "stocks"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ticker = db.Column(db.String(255), primary_key=True)
    date = db.Column(db.Date)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)

    def __repr__(self):
        return '<Stock %r>' % self.ticker

    @staticmethod
    def get_data(tickers, start_date, end_date, freq="D", cols=None):
        '''
        Check if ticker already exists in database. If not, query Tiingo
        '''

        try:

            stock_query = Stock.query.filter(
                Stock.date >= start_date,
                Stock.date <= end_date,
                Stock.ticker.in_(tickers)
            )

            stock_data = pd.read_sql(stock_query.statement, db.session.bind)

            if stock_data.shape[0]:
                stock_data = stock_data.groupby(['date', 'ticker']).last().reset_index()
                if cols is not None and isinstance(cols, list):
                    stock_data = stock_data[cols]

                # TODO: grouper was giving me errors
                # stock_data = stock_data.groupby(pd.Grouper(freq=freq)).last().dropna()

                # TODO: This is really slow and computationally expensive
                retrieved_tickers = stock_data.ticker.unique().tolist()
                missing_tickers = [x for x in tickers if x not in retrieved_tickers]
                print(f"Missing tickers from SQL. Retrieving {len(missing_tickers)} tickers from Tiingo... ")
                tiingo_data = Stock.get_tiingo_data(missing_tickers, start_date, end_date, freq, metric_name=cols)
                stock_data = stock_data.append(tiingo_data)

        except Exception as e:
            # Commenting this out to avoid accidentally pulling a whole bunch of data from Tiingo
            print(f"Stock:get_data - Ran into Exception: {e}. Retrieving from Tiingo...")
            stock_data = Stock.get_tiingo_data(tickers, start_date, end_date, freq, metric_name=cols)
            # stock_data = pd.DataFrame({})

        return stock_data

    @staticmethod
    def get_tiingo_data(tickers, start_date, end_date, freq="D", metric_name=None):

        freq_mapping = {"D" : "daily",
                        "M": "monthly"}

        tiingo_col = ["adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]
        col_mapping = {x: x.strip("adj").lower() for x in tiingo_col}

        freq = "D" if freq not in freq_mapping.keys() else freq

        stock_data = pd.DataFrame({})
        for ticker in tickers:
            try:
                if metric_name is not None:
                    data = client.get_dataframe(ticker,
                                                metric_name=metric_name,
                                                startDate=start_date,
                                                endDate=end_date,
                                                frequency=freq_mapping[freq])
                else:
                    data = client.get_dataframe(ticker,
                                                startDate=start_date,
                                                endDate=end_date,
                                                frequency=freq_mapping[freq])

                data = data[tiingo_col].rename(columns=col_mapping)
                data["ticker"] = ticker
                data = data.reset_index()
                data["id"] = data.index
                data[["open", "close", "high", "low"]] = data[["open", "close", "high", "low"]].apply(lambda x: round(x, 5))
                stock_data = stock_data.append(data)

            except tiingo.restclient.RestClientError:
                print(f"Failed for ticker: {ticker}")

        # Store retrieved stock data to the database
        if stock_data.shape[0]:
            # TODO: Grouper giving me issues
            # stock_data = stock_data.groupby(pd.Grouper(freq=freq)).last().dropna()
            stocks = [Stock(ticker=stock["ticker"], date=stock["date"],
                            open=stock["open"], close=stock["close"],
                            high=stock["high"], low=stock["low"],
                            volume=stock["volume"]) for stock in stock_data.to_dict(orient="rows")]
            db.session.add_all(stocks)
            db.session.commit()

        print(stock_data)
        return stock_data

    @staticmethod
    def get_stock_universe(start_date=constants.START_DATE, end_date=constants.END_DATE):
        return Stock.get_data(constants.STOCK_UNIVERSE, start_date, end_date)


class PortfolioInfo(db.Model):
    __tablename__ = "portfolio_info"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    # Portfolio metadata
    win_philosophy = db.Column(db.Float)
    lose_philosophy = db.Column(db.Float)
    games_philosophy = db.Column(db.Float)
    unknown_philosophy = db.Column(db.Float)
    job_philosophy = db.Column(db.Float)
    monitor_philosophy = db.Column(db.Float)
    name = db.Column(db.String(255))
    time_horizon = db.Column(db.Float)
    cash = db.Column(db.Float)
    holding_constraint = db.Column(db.Float)
    trade_size_constraint = db.Column(db.Float)

    def __repr__(self):
        return '<PortfolioInfo %r>' % self.id

    def get_portfolio_instance(self, user_id, portfolio_name):
        portfolio_instance = self.query.filter_by(user_id=user_id, name=portfolio_name).first()
        return portfolio_instance

    def get_portfolios(self, user_id):
        portfolios = PortfolioInfo.query.filter_by(user_id=user_id)
        return portfolios


    def create_portfolio(self):

        # TODO
        # Get risk tolerance from user input

        # Iniitalize Data
        price_data = Stock.get_stock_universe(constants.START_DATE, constants.END_DATE)
        price_data = price_data[~price_data["close"].isnull()]

        # TODO: Fix rfr
        # rfr = Stock.get_data(constants.RF_RATE, constants.START_DATE, constants.END_DATE).set_index("date")[["close"]].rename(columns={"close": "risk_free"})
        price_data = price_data[["date", "ticker", "close"]].pivot(index=price_data["date"], columns="ticker")["close"]
        original_shape = price_data.shape[0]
        price_data = price_data.dropna(thresh=1000, axis=1)
        print(f"Dropping {original_shape - price_data.shape[0]} entries, {price_data.shape[0]} left. {price_data.shape[1]} stocks")
        rfr = pd.DataFrame({'risk_free': [0.01]*len(price_data.index)}, index = price_data.index)
        data_set = Data(price_data, rfr)
        data_set.set_factor_returns()

        # Initialize Portfolio
        num_stocks = data_set.get_num_stocks()
        port = Portfolio(data_set)

        # Set Up model
        end_date = date.today()
        start_date = end_date - relativedelta(years=3)

        # TODO: These should probably be initialized in constants or smth?
        lookback = 10
        lookahead = 5
        lam = 0.9
        trans_coeff = 0
        holding_coeff = 0
        conf_level = 0.95

        # Define constraints to use
        constr_list = ["no_short", "cardinality", "asset_limit_cardinality"]
        constr_model = Constraints(constr_list)

        cost_model = Costs(trans_coeff, holding_coeff)
        cost_model.replicate_cost_coeff(num_stocks, lookahead)

        opt_model = Model(lam)
        risk_model = Risks("MVO", conf_level)

        regress_weighting = [0, 0.5, 0.5, 0]
        factor_model = FactorModel(lookahead, lookback, regress_weighting)

        back_test_ex = Backtest(start_date, end_date, lookback, lookahead)
        back_test_ex.run(data_set, port, factor_model, opt_model, constr_model, cost_model, risk_model)

        cumu_returns = np.array([x + 1 for x in port.returns]).cumprod()
        portfolio = pd.DataFrame({"date": [start_date] + port.dates, "value": [self.cash] + (cumu_returns * self.cash).tolist(),
                                  "assets": [price_data.columns.tolist() + ["RISK_FREE"] for _ in range(len(port.weights))],
                                  "weights": [x.tolist() for x in port.weights]
                                  })
        portfolio.loc[:, "user_id"] = self.user_id
        portfolio.loc[:, "portfolio_id"] = self.id

        return [PortfolioData(user_id=p['user_id'], portfolio_id=p['portfolio_id'], date=p['date'],
                              assets=p["assets"], weights=p["weights"],
                              value=p['value']) for p in
                portfolio.to_dict(orient="rows")]


class PortfolioData(db.Model):
    __tablename__ = "portfolio_data"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio_info.id'))

    # Time-series data
    date = db.Column(db.Date)
    assets = db.Column(db.ARRAY(db.String(255)))
    weights = db.Column(db.ARRAY(db.Float))
    value = db.Column(db.Float)

    def get_portfolio_data_df(self, user_id, portfolio_id):
        portfolio_data = self.query.filter_by(user_id=user_id, portfolio_id=portfolio_id)
        portfolio_data_df = pd.read_sql(portfolio_data.statement, db.session.bind)
        return portfolio_data_df


db.create_all()  # Create tables in the db if they do not already exist


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
