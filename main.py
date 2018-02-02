from importlib import reload
from pandas import DataFrame
import data_gather
import feature_generation
import learning
import util

reload(data_gather)
reload(learning)
reload(feature_generation)
reload(util)


def test(features):
    result = {}
    for returns, returns_lag, rollmean, rollmean_lag in features:
        idx = util.to_tuple(returns, returns_lag, rollmean, rollmean_lag)
        result[idx] = test2(returns, returns_lag, rollmean, rollmean_lag)

    if output_file is not None:
        output_file.close()

    return DataFrame.from_dict(result).transpose()


def test2(returns, returns_lag, rollmean, rollmean_lag):
    util.print_stats(returns, returns_lag, rollmean, rollmean_lag, file=output_file)

    dataset = create_dataset(returns, returns_lag, rollmean, rollmean_lag)

    result = learning.test(dataset, min_size=1000, step=step)
    util.print_result(result, file=output_file)

    return result


def create_dataset(returns, returns_lag, rollmean, rollmean_lag):
    australia = feature_generation.generate_dataset('axjo', returns, returns_lag, rollmean, rollmean_lag)
    dow_jones = feature_generation.generate_dataset('dji', returns, returns_lag, rollmean, rollmean_lag)
    frankfurt = feature_generation.generate_dataset('gdaxi', returns, returns_lag, rollmean, rollmean_lag)
    hongkong = feature_generation.generate_dataset('hsi', returns, returns_lag, rollmean, rollmean_lag)
    nasdaq = feature_generation.generate_dataset('ixic', returns, returns_lag, rollmean, rollmean_lag)
    nikkei = feature_generation.generate_dataset('n225', returns, returns_lag, rollmean, rollmean_lag)
    paris = feature_generation.generate_dataset('fchi', returns, returns_lag, rollmean, rollmean_lag)
    sp500 = feature_generation.generate_dataset('sp500tr', returns, returns_lag, rollmean, rollmean_lag, b_generate_output=True)

    dataset = feature_generation.join_all([australia, dow_jones, frankfurt, hongkong, nasdaq, nikkei, paris, sp500])

    return dataset


# configurable parameters #
output_file = open('out.txt', 'w')
step = 100
# features = [([1, 2, 3], [1, 2], [2, 3], [])]
features = util.generate_features((3, 3), 9, 2, 0)

result = test(features)

rf = result['rf']

import seaborn

z = result.reset_index()
seaborn.factorplot(x='level_1', y='rf', data=z)
pyplot.show()
