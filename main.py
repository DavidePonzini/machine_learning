from matplotlib import pyplot as plt
from importlib import reload
from pandas import DataFrame
import data_gather
import feature_generation
import learning
import util
import itertools

reload(data_gather)
reload(learning)
reload(feature_generation)
reload(util)


def test(features, feature_tested):
    result = {}
    for feature in features:
        print(feature)
        returns, returns_lag, rollmean, rollmean_lag = feature

        idx = feature[feature_tested][-1]
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


def plot(result, title):
    x = []
    for alg in result.columns:
        x.append(result.index)
        x.append(result[alg])

    plt.plot(*x)
    plt.legend(result.columns)
    plt.title(title)

    plt.show()


def generate_features(returns, returns_lag, rollmean, rollmean_lag):
    return list(itertools.product(returns, returns_lag, rollmean, rollmean_lag))


# configurable parameters #
output_file = open('out.txt', 'w')
step = 1

# best return
# features_returns = generate_features(util.increasing_sequence(9), [[1]], [[1]], [[1]])
# result_returns = test(features_returns, 0)

# best return lag
# features_returns = generate_features(util.increasing_sequence(9), util.increasing_sequence(5), [[1]], [[1]])
# result_returns = test(features_returns, 1)

features_returns = generate_features([util.sequence(3)], [[1]], util.increasing_sequence(5)[1:3], [[1]])
result_returns = test(features_returns, 2)

result_returns.to_csv('3-1-3-1.csv')
plot(result_returns, 'rolling mean')
