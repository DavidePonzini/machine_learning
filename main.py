from importlib import reload
import data_gather
import feature_generation
import learning
import util


reload(data_gather)
reload(learning)
reload(feature_generation)
reload(util)

australia = feature_generation.generate_dataset('axjo')
dow_jones = feature_generation.generate_dataset('dji')
frankfurt = feature_generation.generate_dataset('gdaxi')
hongkong = feature_generation.generate_dataset('hsi')
nasdaq = feature_generation.generate_dataset('ixic')
nikkei = feature_generation.generate_dataset('n225')
paris = feature_generation.generate_dataset('fchi')
sp500 = feature_generation.generate_dataset('sp500tr', True)

dataset = feature_generation.join_all([australia, dow_jones, frankfurt, hongkong, nasdaq, nikkei, paris, sp500])

learning.test_all_with_folds(dataset,
                             folds_no=5,
                             output_file=None,  # open('output.txt', mode='w'),
                             idx_start=11, idx_end=len(dataset) - 1)
