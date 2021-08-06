from deephyper.sklearn.regressor import run as sklearn_run


def load_data():
    from sklearn.datasets import load_boston

    X, y = load_boston(return_X_y=True)
    return X, y


def run(config):
    return sklearn_run(config, load_data)