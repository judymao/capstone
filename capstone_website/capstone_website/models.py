from capstone_website import db, login_manager
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
import tiingo
import pandas_datareader as pdr

tiingo_config = {}
tiingo_config['session'] = True
# TODO: API key should be a constant; maybe store in separate file
tiingo_config['api_key'] = "58245c95b56205705dabecbbfd7e8a346b000bf7"  # StockConstants.API
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

    def getParams(self, ticker, start_date, end_date):
        '''
        Check if ticker already exists in database. If not, query Tiingo
        '''

        try:
            data = client.get_dataframe(ticker,
                                        metric_name="adjClose",
                                        startDate=start_date,
                                        endDate=end_date,
                                        frequency="monthly")  # TODO: turn this to variable
        except tiingo.restclient.RestClientError:
            print(f"Failed for ticker: {ticker}")

        rets = data.pct_change().dropna()  # create return timeseries
        rets.columns = [ticker]

        mu = rets.mean()
        std = rets.std()


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
