import pandas
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


def tune_svm(dataset: pandas.DataFrame, cs, min_size=1000, step=1):
    res = {}

    for c in cs:
        tmp = []
        for i in range(min_size, len(dataset), step):
            print(i, '/', len(dataset), '\tc=', c, sep='')

            xtr, ytr, xts, yts = prepare_dataset(dataset, idx_start=0, idx_end=i)
            cls = SVC(C=c)
            val = _classify(cls, xtr, ytr, xts, yts)
            tmp.append(val)

        res[c] = util.list_avg(tmp)

    return pandas.Series(res)


def test(dataset: pandas.DataFrame, min_size=1000, step=1):
    res_rf = []
    res_knn = []
    res_ada_b = []
    res_svm = []
    res_gtree_b = []

    for i in range(min_size, len(dataset), step):
        print(i)

        xtr, ytr, xts, yts = prepare_dataset(dataset, idx_start=0, idx_end=i)
        res_rf.append(random_forest(xtr, ytr, xts, yts))
        res_knn.append(knn(xtr, ytr, xts, yts))
        res_ada_b.append(adaptive_boosting(xtr, ytr, xts, yts))
        res_gtree_b.append(gradient_tree_boosting(xtr, ytr, xts, yts))
        res_svm.append(svm(xtr, ytr, xts, yts))

    return {
        'rf':      util.list_avg(res_rf),
        'knn':     util.list_avg(res_knn),
        'ada_b':   util.list_avg(res_ada_b),
        'gtree_b': util.list_avg(res_gtree_b),
        'svm':     util.list_avg(res_svm)
    }


def prepare_dataset(dataset, perc_test=0.5, idx_start=0, idx_end=None):
    if not idx_end:
        idx_end = len(dataset)
    dataset = util.trim_dataset(dataset, idx_start, idx_end)

    # start_test = int(len(dataset) * perc_test)

    features_x = dataset.columns[1:-1]
    features_y = dataset.columns[-1]

    x = dataset[features_x]
    y = dataset[features_y]

    xtr = x.iloc[:-1]
    ytr = y.iloc[:-1]

    xts = [x.iloc[-1]]
    yts = [y.iloc[-1]]

    return xtr, ytr, xts, yts


def _classify(classifier, xtr, ytr, xts, yts):
    classifier.fit(xtr, ytr)
    return classifier.score(xts, yts)


def random_forest(xtr, ytr, xts, yts):
    classifier = RandomForestClassifier(n_estimators=100)
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
