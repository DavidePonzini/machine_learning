from matplotlib import pyplot as plt
from importlib import reload
from pandas import DataFrame
import data_gather
import feature_generation
import learning
import util
import itertools
import numpy as np

reload(data_gather)
reload(learning)
reload(feature_generation)
reload(util)


def test_features(features, feature_tested, output_file=None):
    result = {}
    for feature in features:
        returns, returns_lag, rollmean, rollmean_lag = feature

        idx = feature[feature_tested][-1]
        result[idx] = test_feature(returns, returns_lag, rollmean, rollmean_lag, output_file)

    if output_file is not None:
        output_file.close()

    return DataFrame.from_dict(result).transpose()


def test_feature(returns, returns_lag, rollmean, rollmean_lag, output_file=None):
    util.print_stats(returns, returns_lag, rollmean, rollmean_lag, file=output_file)

    dataset = feature_generation.create_dataset(returns, returns_lag, rollmean, rollmean_lag)

    result = learning.test(dataset, min_size=1000, step=step)
    util.print_result(result, file=output_file)

    return result




def generate_features(returns, returns_lag, rollmean, rollmean_lag):
    return list(itertools.product(returns, returns_lag, rollmean, rollmean_lag))


# configurable parameters #
output_file = None  # open('out.txt', 'w')
step = 1

# best return
# features_returns = generate_features(util.increasing_sequence(9), [[1]], [[1]], [[1]])
# result_returns = test_features(features_returns, 0, output_file)
# util.plot(result_returns, 'returns')

# best return lag
# features_returns = generate_features(util.increasing_sequence(9), util.increasing_sequence(5), [[1]], [[1]])
# result_returns = test_features(features_returns, 1, output_file)
# util.plot(result_returns, 'returns lag')

# best rolling mean
# features_returns = generate_features([util.sequence(3)], [[1]], util.increasing_sequence(12)[9:], [[1]])
# result_returns = test_features(features_returns, 2, output_file)
# util.plot(result_returns, 'rolling mean')

# best rolling mean lag
# features_returns = generate_features([util.sequence(3)], [[1]], [util.sequence(2)], util.increasing_sequence(5))
# result_returns = test_features(features_returns, 3, output_file)
# util.plot(result_returns, 'rolling mean lag')

# best parameter for svm
best_dataset = feature_generation.create_dataset(util.sequence(3), [1], util.sequence(2), [])
param_tuning = learning.tune_svm(best_dataset, np.logspace(-3, 3, 20), step=1)

plt.plot(param_tuning)
plt.xscale('log')
plt.xlabel('c')
plt.ylabel('error')
plt.show()
