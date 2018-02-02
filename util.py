from pandas import DataFrame
import itertools
from matplotlib import pyplot

def trim_dataset(dataset: DataFrame, idx_start: int, idx_end: int):
    return dataset.iloc[idx_start:idx_end]


def list_avg(l: []):
    if len(l) == 0:
        return None
    return sum(l) / len(l)


def increasing_sequence(max_value: int, min_value=1):
    if max_value < min_value:
        return [[]]

    res = []

    for i in range(min_value, max_value + 1):
        res.append(sequence(start=min_value, end=i))

    return res


def sequence(end, start=1):
    return list(range(start, end + 1))


def generate_features(returns_max: int, returns_lag_max: int, rollmean_max: int, rollmean_lag_max: int):
    returns = increasing_sequence(returns_max[1], returns_max[0])
    returns_lag = [sequence(returns_lag_max)]
    rollmean = increasing_sequence(rollmean_max, 2)
    rollmean_lag = [sequence(rollmean_lag_max)]

    return list(itertools.product(returns, returns_lag, rollmean, rollmean_lag))


def print_stats(returns, returns_lag, rollmean, rollmean_lag, file=None):
    print('returns:', returns, file=file)
    print('returns_lag:', returns_lag, file=file)
    print('rollmean:', rollmean, file=file)
    print('rollmean_lag:', rollmean_lag, file=file)


def print_result(result, file=None):
    print('\tk-nearest_neighbor:   ', result['knn'], file=file)
    print('\trandom_forest:        ', result['rf'], file=file)
    print('\tsupport_vect_machines:', result['svm'], file=file)
    print('\tadaptive_boost:       ', result['ada_b'], file=file)
    print('\tgrad_tree_boost:      ', result['gtree_b'], file=file)
    print(file=file)


def to_tuple(*args):
    res = []
    for arg in args:
        if len(arg) is 0:
            res.append(0)
        else:
            res.append(arg[-1])

    return tuple(res)


def plot(series):
    x = [i[0] for i in list(series.index)]
    y = list(series.values)

    pyplot.plot(x, y)
    pyplot.show()
