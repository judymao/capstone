'''
For cleaning up the databases
'''


from datetime import datetime

from capstone_website.models import Stock
from capstone_website.src.constants import Constants


def clean():
    constants = Constants()
    end_time = datetime.now()
    start_time = datetime(end_time.year - 14, end_time.month, end_time.day)
    print(f"Retrieving data from {start_time} to {end_time}...")

    stocks = Stock.get_data(constants.STOCK_UNIVERSE, start_time, end_time)


if __name__ == "__main__":
    clean()