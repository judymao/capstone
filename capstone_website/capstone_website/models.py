from capstone_website import db, login_manager, client, quandl_api, alpha_vantage_api
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import tiingo
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
import dateutil
import pandas as pd
import numpy as np
import quandl
from alpha_vantage.timeseries import TimeSeries
from scipy import stats

from capstone_website.src.constants import Constants
from capstone_website.src.data import Data
from capstone_website.src.portfolio import Portfolio
from capstone_website.src.factor_models import FactorModel
from capstone_website.src.livetest import Livetest


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
                if cols is not None and isinstance(cols, list):
                    stock_data = stock_data[cols]
                stock_data = stock_data[(stock_data["date"] >= start_date) & (stock_data["date"] <= end_date)]
                stock_data = stock_data.groupby(["date", "ticker"]).last().reset_index()
                # TODO: grouper was giving me errors
                # stock_data = stock_data.groupby(pd.Grouper(freq=freq)).last().dropna()

        except Exception as e:
            print(f"Stock:get_data - Ran into Exception")
            stock_data = pd.DataFrame({})

        return stock_data

    @staticmethod
    def get_stock_data(tickers, start_date, end_date, freq="D", cols=None):
        '''
        Check if ticker already exists in database. If not, query Tiingo
        '''

        stock_data = Stock.get_data(tickers, start_date, end_date, freq, cols)
        if len(stock_data["ticker"].unique()) != len(tickers):

            retrieved_tickers = stock_data.ticker.unique().tolist()
            missing_tickers = [x for x in tickers if x not in retrieved_tickers]
            tiingo_data = Stock.get_tiingo_data(missing_tickers, start_date, end_date, freq, metric_name=cols)
            stock_data = stock_data.append(tiingo_data)

        return stock_data

    @staticmethod
    def get_tiingo_data(tickers, start_date, end_date, freq="D", metric_name=None):

        print(f"Getting data for {len(tickers)} tickers")
        freq_mapping = {"D" : "daily",
                        "M": "monthly"}

        tiingo_col = ["adjOpen", "adjHigh", "adjLow", "adjClose"]
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
                # data["id"] = data.index
                data[["open", "close", "high", "low"]] = data[["open", "close", "high", "low"]].apply(lambda x: round(x, 5))
                stock_data = stock_data.append(data)
                print(f"Retrieved Tiingo data for ticker {ticker} ... {data.shape[0]} entries")

            except tiingo.restclient.RestClientError:
                print(f"Failed for ticker: {ticker}")

        # Store retrieved stock data to the database
        if stock_data.shape[0]:
            # TODO: Grouper giving me issues
            # stock_data = stock_data.groupby(pd.Grouper(freq=freq)).last().dropna()
            stocks = [Stock(ticker=stock["ticker"], date=stock["date"],
                            open=stock["open"], close=stock["close"],
                            high=stock["high"], low=stock["low"]) for stock in stock_data.to_dict(orient="rows")]
            print(f"Storing retrieved data for {len(stocks)} tickers into database...")
            db.session.bulk_insert_mappings(Stock, stock_data.to_dict(orient="rows"))
            # db.session.add_all(stocks)
            db.session.commit()

        return stock_data


    @staticmethod
    def get_stock_universe(start_date, end_date):
        return Stock.get_stock_data(constants.STOCK_UNIVERSE, start_date, end_date)

    @staticmethod
    def get_risk_free(start_date, end_date):
        ten_year_df = Stock.get_data([constants.RF_RATE], start_date, end_date)
        if not ten_year_df.shape[0]:
            print(f"Risk free rate was not in database. Retrieving from Quandl...")
            ten_year = quandl.get("USTREASURY/YIELD", authtoken=quandl_api)["10 YR"]
            ten_year_df = pd.DataFrame(ten_year[(ten_year.index >=
                                                 pd.to_datetime(start_date)) &
                                                (ten_year.index <=
                                                 pd.to_datetime(end_date))]).reset_index()
            ten_year_df = ten_year_df.rename(columns={"Date": "date"})
            ten_year_df["date"] = ten_year_df["date"].astype(object).apply(lambda x: x.date())
            ten_year_df["10 YR"] = ten_year_df["10 YR"] / 100
            ten_year_df["ticker"] = constants.RF_RATE
            stock = [Stock(ticker=stock["ticker"], date=stock["date"], close=stock["10 YR"]) for stock in ten_year_df.to_dict(orient="rows")]
            db.session.add_all(stock)
            # db.session.bulk_insert_mappings(Stock, stock)
            db.session.commit()
        return ten_year_df

    @staticmethod
    def get_etf(etf, start_date, end_date):
        spy_df = Stock.get_data([etf], start_date, end_date)
        if not spy_df.shape[0]:
            print(f"{etf} was not in database. Retrieving from AlphaVantage...")
            ts = TimeSeries(key=alpha_vantage_api, indexing_type='date')
            spy_df = pd.DataFrame(ts.get_daily_adjusted(etf, outputsize="full")[0])
            spy_df.index = pd.Series(spy_df.index).apply(lambda x: x.split(" ")[1])
            spy_df = spy_df.transpose().reset_index().rename(columns={"index": "date"})
            spy_df["date"] = spy_df["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
            spy_df = spy_df[(spy_df["date"] >= start_date) & (spy_df["date"] <= end_date)]
            spy_df["ticker"] = etf
            stock = [Stock(ticker=stock["ticker"], date=stock["date"], close=stock["close"],
                           high=stock["high"], low=stock["low"], open=stock["open"]) for stock in
                     spy_df.to_dict(orient="rows")]
            db.session.add_all(stock)
            db.session.commit()
        return spy_df


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
    returns = db.Column(db.Float)
    volatility = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    risk_appetite = db.Column(db.String(255))

    def __repr__(self):
        return '<PortfolioInfo %r>' % self.id

    def get_portfolio_instance(self, user_id, portfolio_name):
        portfolio_instance = self.query.filter_by(user_id=user_id, name=portfolio_name).first()
        return portfolio_instance

    def get_portfolios(self, user_id):
        portfolios = PortfolioInfo.query.filter_by(user_id=user_id)
        return portfolios

    def get_baseline_portfolios(self, risk_appetite):
        portfolio = self.query.filter_by(name="Baseline_" + risk_appetite).first()
        return portfolio


    def create_portfolio(self):

        # Iniitalize Data
        price_data = Stock.get_stock_universe(constants.START_DATE, constants.END_DATE)
        price_data = price_data[~price_data["close"].isnull()]

        rfr = Stock.get_risk_free(constants.START_DATE, constants.END_DATE).set_index("date")[["close"]].rename(columns={"close": "risk_free"})
        rfr["risk_free"] = (1 + (rfr["risk_free"])) ** (1 / 12) - 1
        price_data = price_data[["date", "ticker", "close"]].pivot(index="date", columns="ticker", values="close")
        original_shape = price_data.shape[0]
        price_data = price_data.dropna(thresh=3000, axis=1)
        print(f"Dropping {original_shape - price_data.shape[0]} entries, {price_data.shape[0]} left. {price_data.shape[1]} stocks")

        data = Data(price_data, rfr)
        FF_data = Data(price_data, rfr, factor_type="FF")

        # Set up Portfolio
        port = Portfolio(data)

        # Set up Static Variables
        end_date = "2020-11-26"
        # start_date = "2010-01-31"
        string_date = datetime.strptime(end_date, "%Y-%m-%d")
        string_date_minus = string_date - dateutil.relativedelta.relativedelta(years=self.time_horizon)
        start_date = string_date_minus.strftime('%Y-%m-%d')
        lookback = 12
        lookahead = 1

        # Map user survey to risk appetite
        high_risk_ret = 0.15
        medium_risk_ret = 0.10
        low_risk_ret = 0.05

        risk_appetite = (self.win_philosophy + self.lose_philosophy + self.games_philosophy + self.unknown_philosophy + self.job_philosophy + self.monitor_philosophy)
        if risk_appetite < 4:
            return_goal = low_risk_ret
            self.risk_appetite = "Low"
        elif risk_appetite < 9:
            return_goal = medium_risk_ret
            self.risk_appetite = "Medium"
        else:
            return_goal = high_risk_ret
            self.risk_appetite = "High"

        baseline_portfolio = self.get_baseline_portfolios(self.risk_appetite)

        # If baseline exists, inherit values from this baseline portfolio
        # if baseline_portfolio.sharpe_ratio is not None:
        if baseline_portfolio is not None:
                print(f"Identified baseline portfolio for risk appetite: {self.risk_appetite}")
                portfolio_data = PortfolioData()
                baseline_portfolio_df = portfolio_data.get_portfolio_data_df(baseline_portfolio.user_id, baseline_portfolio.id)
                baseline_portfolio_df.loc[:, "user_id"] = self.user_id
                baseline_portfolio_df.loc[:, "portfolio_id"] = self.id
                portfolio = baseline_portfolio_df[baseline_portfolio_df["date"] > datetime.strptime(start_date, "%Y-%m-%d").date()]
                portfolio["returns"] = portfolio["value"].pct_change()

                self.returns = (portfolio.iloc[-1]["value"] - portfolio.iloc[0]["value"]) / portfolio.iloc[0]["value"]

                sigma = np.std(portfolio["returns"] - np.array(rfr["risk_free"].iloc[-1])) # this is not getting correct rate over time
                sharpe_ratio = (stats.gmean(portfolio["returns"].dropna() - np.array(rfr["risk_free"].iloc[-1]) + 1, axis=0) - 1) / sigma
                annual_sharpe = sharpe_ratio * (np.sqrt(12))

                self.volatility = sigma * np.sqrt(12)
                self.sharpe_ratio = annual_sharpe

                portfolio["value"] = (self.cash / portfolio.iloc[0]["value"]) * portfolio["value"]

                return [PortfolioData(user_id=p['user_id'], portfolio_id=p['portfolio_id'], date=p['date'],
                                      assets=p["assets"], weights=p["weights"],
                                      value=p['value']) for p in
                        portfolio.to_dict(orient="rows")]

        else:
            regress_weighting = [0, 0, 0.25, 0.75]
            factor_model = FactorModel(lookahead, lookback, regress_weighting)

            live_test = Livetest(start_date, end_date)
            sharpe_ratio = live_test.run(data, FF_data, port, return_goal, factor_model)

            cumu_returns = np.array([x + 1 for x in port.returns]).cumprod()
            portfolio = pd.DataFrame({"date": [start_date] + port.dates, "value": [self.cash] + (cumu_returns * self.cash).tolist(),
                                      "assets": [price_data.columns.tolist() + ["RISK_FREE"] for _ in range(len(port.weights))],
                                      "weights": [x.tolist() for x in port.weights]
                                      })
            portfolio.loc[:, "user_id"] = self.user_id
            portfolio.loc[:, "portfolio_id"] = self.id

            self.returns = cumu_returns[-1] - 1
            self.volatility = np.std(port.returns - np.array(rfr["risk_free"].iloc[-1])) * np.sqrt(12)
            self.sharpe_ratio = sharpe_ratio

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
