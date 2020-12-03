
import pandas as pd
import numpy as np
import urllib.request
import zipfile
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import skew, pearson3


class Data:
    # Anything Data Related
    def __init__(self, stock_prices, risk_free, universe=None, factor_type='PCA', period='M'):
        # TO-DO: Add initialization of market cap

        if not universe:
            universe = stock_prices.columns

        if type(universe[0]) == int:
            self.stock_prices = stock_prices.iloc[:, universe]

        else:
            self.stock_prices = stock_prices[universe]

        self.risk_free = risk_free
        self.risk_free.index = pd.to_datetime(self.risk_free.index)
        self.risk_free = self.risk_free.resample(period).last()
        self.stock_prices.index = pd.to_datetime(self.stock_prices.index)
        self.stock_returns = self.get_stock_returns(period)
        self.factor_returns = self.get_factor_returns(factor_type)

        return

    def get_stock_returns(self, period='M'):
        price = self.stock_prices.resample(period).last()

        # Calculate the percent change
        ret_data = price.pct_change()[1:]

        # Convert from series to dataframe
        ret_data = pd.DataFrame(ret_data)

        return ret_data

    def get_factor_returns(self, factor_type='PCA', period='M'):
        if factor_type == 'CAPM':

            return self.get_CAPM_returns(period)

        elif factor_type == 'FF':

            return self.get_FF_returns(period)

        elif factor_type == 'Carhart':

            return self.get_Carhart_returns(period)

        elif factor_type == 'PCA':

            return self.get_PCA_returns(period)

        else:
            print("Invalid input: Please select one of the following factor types: CAPM, FF, Carhart or PCA.")

        return

    def get_FF_returns(self, period='M'):
        ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
        # Download the file and save it
        urllib.request.urlretrieve(ff_url, 'fama_french.zip')
        zip_file = zipfile.ZipFile('fama_french.zip', 'r')
        # Extact the file data
        zip_file.extractall()
        zip_file.close()
        ff_factors = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows=3, index_col=0)
        # Skip null rows
        ff_row = ff_factors.isnull().any(1).to_numpy().nonzero()[0][0]

        # Read the csv file again with skipped rows
        ff_factors = pd.read_csv('F-F_Research_Data_Factors.csv', skiprows=3, nrows=ff_row, index_col=0)

        # Format the date index
        ff_factors.index = pd.to_datetime(ff_factors.index, format='%Y%m')

        # Format dates to end of month
        ff_factors.index = ff_factors.index + pd.offsets.MonthEnd()

        # Resample the data to correct frequency
        ff_factors = ff_factors.resample(period).last()

        # Convert from percent to decimal
        ff_factors = ff_factors.apply(lambda x: x / 100)

        return ff_factors

    def get_CAPM_returns(self, period='M'):
        ff_factors = self.get_FF_returns(period)

        # Remove the unnecessary factors
        capm_factors = ff_factors.iloc[:, 0]

        return capm_factors

    def get_Carhart_returns(self, period='M'):
        ff_factors = self.get_FF_returns(period)

        # Get the momentum factor
        momentum_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"

        # Download the file and save it
        urllib.request.urlretrieve(momentum_url, 'momentum.zip')
        zip_file = zipfile.ZipFile('momentum.zip', 'r')

        # Extact the file data
        zip_file.extractall()
        zip_file.close()

        momentum_factor = pd.read_csv('F-F_Momentum_Factor.csv', skiprows=13, index_col=0)

        # Skip null rows
        row = momentum_factor.isnull().any(1).to_numpy().nonzero()[0][0]

        # Read the csv file again with skipped rows
        momentum_factor = pd.read_csv('F-F_Momentum_Factor.csv', skiprows=13, nrows=row, index_col=0)

        # Format the date index
        momentum_factor.index = pd.to_datetime(momentum_factor.index, format='%Y%m')

        # Format dates to end of month
        momentum_factor.index = momentum_factor.index + pd.offsets.MonthEnd()

        # Resample the data to correct frequency
        momentum_factor = momentum_factor.resample(period).last()

        # Convert from percent to decimal
        momentum_factor = momentum_factor.apply(lambda x: x / 100)

        # Combine to create the carhart_factors
        carhart_factors = pd.concat([ff_factors, momentum_factor], axis=1).dropna()

        return carhart_factors

    def get_PCA_returns(self, period='M'):
        exRets = self.get_stock_returns(period="D")
        num_stocks = len(exRets.columns)
        returns_mat = exRets.to_numpy()
        n_dates = returns_mat.shape[0]
        n_assets = returns_mat.shape[1]

        demeaned = (returns_mat - returns_mat.mean(axis=0)).transpose()
        sigma = 1 / (n_dates - 1) * np.matmul(demeaned, demeaned.transpose())
        eigval, eigvec = np.linalg.eig(sigma)

        principal_components = np.matmul(eigvec.transpose(), demeaned).transpose()
        pca_factors = np.real(principal_components[:, 0:10])

        pca_df = pd.DataFrame(pca_factors, index=exRets.index, columns=[str(i) for i in range(10)])
        pca_df = pca_df.resample(period).last()

        return pca_df

    def get_index_from_date(self, date_index_df, date):
        return date_index_df.index.get_loc(date)

    def get_lookback_data(self, date_index_df, date, lookback):
        end_idx = self.get_index_from_date(date_index_df, date)
        return date_index_df.iloc[end_idx - lookback:end_idx]

    def get_num_stocks(self):
        return len(self.stock_returns.columns)
