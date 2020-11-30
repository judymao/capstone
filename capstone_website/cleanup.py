'''
For cleaning up the databases
'''


from datetime import datetime

from capstone_website.models import Stock
from capstone_website.src.constants import Constants


def clean():

    end_time = datetime.now()
    start_time = datetime(end_time.year - 14, end_time.month, end_time.day)
    print(f"Retrieving data from {start_time} to {end_time}...")

    print(f"Retrieving stock data...")
    stocks = Stock.get_stock_universe(start_time, end_time)
    print(f"\tRetrieved {len(stocks['ticker'].unique().tolist())} stocks, {stocks.shape[0]:,.0f} entries")

    print(f"Retrieving risk free rate...")
    risk_free = Stock.get_risk_free(start_time, end_time)
    print(f"\tRetrieved {len(risk_free['ticker'].unique().tolist())} risk free rate")

    print(f"Done cleanup!")

if __name__ == "__main__":
    clean()