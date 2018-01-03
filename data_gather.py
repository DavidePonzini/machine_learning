import pandas
from pandas_datareader import data as pd_reader
from datetime import datetime
from os import path


start = datetime(2010, 1, 1)
end = datetime.now()

_data_dir = 'data'


def get_file_path(symbol):
    return path.join(_data_dir, symbol + '.csv')


def download_data(symbol):
    data = pd_reader.get_data_yahoo(symbol, start, end)

    data.to_csv(get_file_path(symbol))


def _has_increased(col1, col2):
    return col1.combine(col2, lambda x, y: 1 if x > y else -1)


def _increase(col1, col2):
    return col1.combine(col2, lambda x, y: (y - x) / x)


def read_data(symbol):
    data = pandas.read_csv(get_file_path(symbol))

    # data['open_ch'] = data['Open'].shift(0).pct_change()  # shift(1) if trying to predict open price
    # data['close_ch'] = data['Close'].shift(0).pct_change()
    # data['high_ch'] = data['High'].shift(1).pct_change()
    # data['low_ch'] = data['Low'].shift(1).pct_change()
    # data['incr_day'] = data['Close'].pct_change()

    # del data['Open']
    # del data['Close']
    del data['Adj Close']
    # del data['High']
    # del data['Low']
    del data['Volume']

    data.columns = data.columns + '_' + symbol
    data.columns.values[0] = 'Date'

    return data[11:]


