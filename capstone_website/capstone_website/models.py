from capstone_website import db, login_manager
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import tiingo
import pandas_datareader as pdr
from datetime import date
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import chart_studio.plotly as py

# TODO: This is really hacky initialization
import chart_studio
chart_studio.tools.set_credentials_file(username='marycapstone', api_key='lLReEJuDrPeBrZCBzpMr')

tiingo_config = {}
tiingo_config['session'] = True
# TODO: API key should be a constant; maybe store in separate file
tiingo_config['api_key'] = "57faeaf57f08c983e03aee6f91ffc72ba2c40a55"  # StockConstants.API
client = tiingo.TiingoClient(tiingo_config)


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

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(255))
    date = db.Column(db.Date)
    open = db.Column(db.Float)
    high = db.Column(db.Float)
    low = db.Column(db.Float)
    close = db.Column(db.Float)
    volume = db.Column(db.Integer)

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
                #TODO: grouper was giving me errors
                # stock_data = stock_data.groupby(pd.Grouper(freq=freq)).last().dropna()

            retrieved_tickers = stock_data.ticker.unique().tolist()
            missing_tickers = [x for x in tickers if x not in retrieved_tickers]
            tiingo_data = Stock.get_tiingo_data(missing_tickers, start_date, end_date, freq, metric_name=cols)
            stock_data = stock_data.append(tiingo_data)

        except Exception as e:
            print(f"Stock:get_data - Ran into Exception: {e}. Retrieving from Tiingo...")
            stock_data = Stock.get_tiingo_data(tickers, start_date, end_date, freq, metric_name=cols)

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

class PortfolioInfo(db.Model):
    __tablename__ = "portfolio_info"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    # Portfolio metadata
    protect_portfolio = db.Column(db.String(255))
    inv_philosophy = db.Column(db.String(255))
    next_expenditure = db.Column(db.String(255))
    name = db.Column(db.String(255))
    time_horizon = db.Column(db.Float)
    holding_constraint = db.Column(db.Float)
    trade_size_constraint = db.Column(db.Float)


    def create_portfolio(self):

        # this is where the optimization and factor model can probably come in

        # query Stock object to get stocks
        # random_stock = Stock.query

        # Create a random portfolio
        # This code is garbage but will be replaced so whatevs I guess

        start_date = date(2019, 11, 10)
        tickers = ["GOOGL", "AAPL"]
        num_assets = len(tickers)

        # stock_query = Stock.query.filter(Stock.date >= start_date)
        # stock_data = pd.read_sql(stock_query.statement, db.session.bind)
        stock_data = Stock.get_data(tickers=tickers, start_date=start_date, end_date=date.today())

        if stock_data.shape[0]:
            # Only get close data and aggregate by date
            # Caution: date formats MIGHT beself different since it's datetime, not date
            stock_data = stock_data[["ticker", "date", "close"]]
            portfolio = pd.DataFrame({"assets": stock_data.groupby("date")["ticker"].unique()}).reset_index()
            # TODO: currently ignoring dates where inconsistent number of assets
            portfolio.loc[:, "close"] = stock_data.groupby("date")["close"].unique().values
            portfolio = portfolio[portfolio["assets"].apply(lambda row: len(row)) > 1]
            portfolio["weights"] = [[1/num_assets for i in range(num_assets)] for x in range(portfolio.shape[0])]
            portfolio.loc[:, "value"] = [np.dot(np.array(portfolio.close.iloc[x]), np.array(portfolio.weights.iloc[x])) for x in range(portfolio.shape[0])]
            portfolio = portfolio.drop("close", axis=1)
            portfolio.loc[:, "user_id"] = self.user_id
            portfolio.loc[:, "portfolio_id"] = self.id

            # Render a graph and return the URL
            fig = go.Figure(data=go.Scatter(x=portfolio["date"], y=portfolio["value"], mode="lines", name="Portfolio Value"))
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='Portfolio Value')
            portfolio_graph_url = py.plot(fig, filename="portfolio_value", auto_open=False, )
            # print(portfolio_graph_url)

            # TODO
            # Render a table of portfolio stats
            # portfolio_table =

            return portfolio_graph_url, [PortfolioData(user_id=p['user_id'], portfolio_id=p['portfolio_id'], date=p['date'],
                                                       assets=p['assets'], weights=p['weights'], value=p['value']) for p in portfolio.to_dict(orient="rows")]



    # def backtest(self):



class PortfolioData(db.Model):
    __tablename__ = "portfolio_data"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))
    portfolio_id = db.Column(db.Integer, db.ForeignKey('portfolio_info.id'))

    # Time-series data
    date = db.Column(db.Date)
    assets = db.Column(db.ARRAY(db.String(255)))
    weights = db.Column(db.ARRAY(db.Float))
    value = db.Column(db.Integer)




db.create_all() # Create tables in the db if they do not already exist


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
