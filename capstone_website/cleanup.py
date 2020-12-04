'''
For cleaning up the databases
'''


from datetime import datetime

from capstone_website.models import Stock, PortfolioInfo, PortfolioData
from capstone_website.src.constants import Constants

from capstone_website import db

def clean():

    constants = Constants()

    # end_time = datetime.now().date()
    # start_time = datetime(end_time.year - 14, end_time.month, end_time.day).date()
    # print(f"Retrieving data from {start_time} to {end_time}...\n")
    #
    # # TODO: Maybe also remove stocks that are not part of the stock universe to clean up the database
    # print(f"Retrieving stock data...")
    # stocks = Stock.get_stock_universe(start_time, end_time)
    # print(f"\tRetrieved {len(stocks['ticker'].unique().tolist())} stocks, {stocks.shape[0]:,.0f} entries")
    #
    # print(f"Retrieving risk free rate...")
    # risk_free = Stock.get_risk_free(start_time, end_time)
    # print(f"\tRetrieved {len(risk_free['ticker'].unique().tolist())} risk free rate with {risk_free.shape[0]:,.0f} entries")
    #
    # print(f"Retrieving SPY...")
    # spy = Stock.get_etf(constants.SPY, start_time, end_time)
    # print(f"\tRetrieved {len(spy['ticker'].unique().tolist())} SPY with {spy.shape[0]:,.0f} entries")
    #
    # print(f"Retrieving Dow Jones ETF...")
    # etf = Stock.get_etf(constants.DOW_JONES, start_time, end_time)
    # print(f"\tRetrieved {len(etf['ticker'].unique().tolist())} DIA with {etf.shape[0]:,.0f} entries")


    print(f"Removing empty portfolios...")
    portfolios = PortfolioInfo.query.filter_by().all()
    missing_data = [portfolio.id for portfolio in portfolios if portfolio.risk_appetite is None or portfolio.returns is None]
    print(f"Removing {len(missing_data)} missing portfolios")

    for id in missing_data:
        PortfolioData.query.filter_by(portfolio_id=id).delete()
        PortfolioInfo.query.filter_by(id=id).delete()
        db.session.commit()
    print(f"Removed {len(missing_data)} missing portfolios")
    print(f"Done cleanup!")

if __name__ == "__main__":
    clean()