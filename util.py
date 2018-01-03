from pandas import DataFrame


def trim_dataset(dataset: DataFrame, idx_start: int, idx_end: int):
    # dataset = dataset[dataset.index >= idx_start]
    # dataset = dataset[dataset.index < idx_end]

    return dataset.iloc[idx_start:idx_end]
