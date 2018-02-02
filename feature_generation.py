import pandas
import data_gather
from sklearn import preprocessing


def generate_dataset(symbol: str, returns: [], returns_lag: [], rollmean: [], rollmean_lag: [], b_generate_output: bool=False):
    ds = data_gather.read_data(symbol)

    if len(returns) is not 0:
        _generate_features_return(ds, ['Close_' + symbol], returns)
    if len(returns_lag) is not 0:
        _generate_features_lag(ds, returns_lag, ['Close_' + symbol + '_return_'], returns)
    if len(rollmean) is not 0:
        _generate_features_rolling_mean(ds, ['Close_' + symbol], rollmean)
    if len(rollmean_lag) is not 0:
        _generate_features_lag(ds, rollmean_lag, ['Close_' + symbol + '_rollmean_'], rollmean)

    if b_generate_output:
        _generate_features_output(ds, 'Close_' + symbol)

    del ds['Close_' + symbol]
    del ds['Open_' + symbol]
    del ds['High_' + symbol]
    del ds['Low_' + symbol]

    return ds


def join_all(dataframes: [pandas.DataFrame]):
    df = dataframes[0]

    for dataframe in dataframes[1:]:
        df = _join(df, dataframe)

    return df.dropna()


def _join(left: pandas.DataFrame, right: pandas.DataFrame):
    return pandas.merge(left, right, how='inner', on='Date')


def _generate_features_lag(dataset: pandas.DataFrame, lags: [int], cols: [str], deltas: [int]):
    for col in cols:
        for lag in lags:
            for delta in deltas:
                col_name = str(col) + str(delta)
                name = col_name + '_lag_' + str(lag)
                dataset[name] = dataset[col_name].shift(lag)


def _generate_features_rolling_mean(dataset: pandas.DataFrame, cols: [str], deltas: [int]):
    for col in cols:
        for delta in deltas:
            returns = dataset[col].pct_change()
            name = col + '_rollmean_' + str(delta)
            dataset[name] = pandas.Series(returns).rolling(window=delta, center=False).mean()


def _generate_features_return(dataset: pandas.DataFrame, cols: [str], deltas: [int]):
    for col in cols:
        for delta in deltas:
            name = col + '_return_' + str(delta)
            dataset[name] = dataset[col].pct_change(delta)


def _generate_features_output(dataset: pandas.DataFrame, col_name):
    dataset['out'] = dataset[col_name].shift(1) < dataset[col_name]
    dataset['out'] = dataset['out'].shift(-1)

    idx = dataset.index[-1]

    # Temporarily disable false positive warning
    #     (I don't care about writes making it back to the original dataframe,
    #      since I'm overwriting the reference to it)
    pandas.options.mode.chained_assignment = None

    dataset['out'][idx] = False

    # Restore normal warning behaviour
    pandas.options.mode.chained_assignment = 'warn'

    label_encoder = preprocessing.LabelEncoder()
    dataset['out'] = label_encoder.fit(dataset['out']).transform(dataset['out'])
