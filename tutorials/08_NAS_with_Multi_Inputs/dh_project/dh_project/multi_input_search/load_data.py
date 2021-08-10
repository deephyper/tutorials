import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_data():
    """Here we are loading data that has multiple inputs for the same output.
    Our goal is to not make ONE tensor with all inputs but have separate input tensors.
    """

    data = np.load(os.path.join(HERE, "data.npz")) # Data from the Brannin function

    xtrain = data["xtrain"]  # Independent variables (input tensor 1)
    ytrain_lf = data["ytrain_lf"]  # Low fidelity variable (input tensor 2)
    ytrain_mf = data["ytrain_mf"]  # Medium fidelity variable (input tensor 3)
    ytrain_hf = data["ytrain_hf"]  # High fidelity variable (output tensor)

    xtest = data["xtest"]
    ytest_lf = data["ytest_lf"]
    ytest_mf = data["ytest_mf"]
    ytest_hf = data["ytest_hf"]

    return ([xtrain, ytrain_lf, ytrain_mf], ytrain_hf), (
        [xtest, ytest_lf, ytest_mf],
        ytest_hf,
    )


if __name__ == "__main__":
    load_data()