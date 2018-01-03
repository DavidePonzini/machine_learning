from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from data_gather import read_data
from learning import prepare_dataset


def get_ranking(name):
    dataset = read_data(name)

    rfe = RFE(LogisticRegression(), 5)

    xtr, ytr, xts, yts = prepare_dataset(dataset)

    rfe.fit(xtr, ytr)

    return rfe.ranking_
