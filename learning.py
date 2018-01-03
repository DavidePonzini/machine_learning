import pandas
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
import util


def _split_dataset_in_folds(dataset: pandas.DataFrame, folds_no: int, output_file=None):
    fold_len = len(dataset) // folds_no
    fold_end = fold_len

    for i in range(0, folds_no):
        ds = dataset.iloc[:fold_end]

        print('\n', 'fold', i + 1,
              'from', ds.iloc[0]['Date'],
              'to', ds.iloc[-1]['Date'],
              'no_elems', len(ds),
              sep=' ', file=output_file)

        yield ds

        fold_end = fold_end + fold_len


def test_all_with_folds(dataset: pandas.DataFrame, folds_no: int, idx_start=0, idx_end=None, output_file=None):
    dataset = util.trim_dataset(dataset, idx_start, idx_end)

    for ds in _split_dataset_in_folds(dataset, folds_no=folds_no, output_file=output_file):
        test_all(ds, output_file=output_file)


def test_all(dataset: pandas.DataFrame, perc=0.5, idx_start=0, idx_end=None, output_file=None):
    xtr, ytr, xts, yts = prepare_dataset(dataset, perc, idx_start, idx_end)

    print('rand_forest:\t', random_forest(xtr, ytr, xts, yts), file=output_file)
    print('k-near_neigh:\t', knn(xtr, ytr, xts, yts), file=output_file)
    print('supp_vect_m:\t', svm(xtr, ytr, xts, yts), file=output_file)
    print('ada_boost:\t\t', adaptive_boosting(xtr, ytr, xts, yts), file=output_file)
    print('gtree_boost:\t', gradient_tree_boosting(xtr, ytr, xts, yts), file=output_file)


def prepare_dataset(dataset, perc_test=0.5, idx_start=0, idx_end=None):
    if not idx_end:
        idx_end = len(dataset)
    dataset = util.trim_dataset(dataset, idx_start, idx_end)

    start_test = int(len(dataset) * perc_test)

    features_x = dataset.columns[1:-1]
    features_y = dataset.columns[-1]

    label_encoder = preprocessing.LabelEncoder()
    dataset[features_y] = label_encoder.fit(dataset[features_y]).transform(dataset[features_y])

    x = dataset[features_x]
    y = dataset[features_y]

    xtr = x.iloc[:start_test]
    ytr = y.iloc[:start_test]

    xts = x.iloc[start_test:]
    yts = y.iloc[start_test:]

    return xtr, ytr, xts, yts


def _classify(classifier, xtr, ytr, xts, yts):
    classifier.fit(xtr, ytr)
    return classifier.score(xts, yts)


def random_forest(xtr, ytr, xts, yts):
    classifier = RandomForestClassifier(n_estimators=1000)
    return _classify(classifier, xtr, ytr, xts, yts)


def knn(xtr, ytr, xts, yts):
    classifier = KNeighborsClassifier()

    return _classify(classifier, xtr, ytr, xts, yts)


def svm(xtr, ytr, xts, yts):
    classifier = SVC()

    return _classify(classifier, xtr, ytr, xts, yts)


def adaptive_boosting(xtr, ytr, xts, yts):
    classifier = AdaBoostClassifier()

    return _classify(classifier, xtr, ytr, xts, yts)


def gradient_tree_boosting(xtr, ytr, xts, yts):
    classifier = GradientBoostingClassifier(n_estimators=1000)

    return _classify(classifier, xtr, ytr, xts, yts)
