import numpy as np
import scipy.sparse as sp


def convert_data(X, E, MAX_ATOM, MAX_EDGE, N_FEAT, E_FEAT):
    """Convert an X(node feature), E(edge feature) to desired input format for GNN.

    Args:
        X (np.array): node features.
        E (np.array): edge features.
        MAX_ATOM (int): the maximum number of atoms zero-padding to.
        MAX_EDGE (int): the maximum number of edges zero-padding to.
        N_FEAT (int): the number of node features.
        E_FEAT (int): the number of edge features.

    Returns:
        Xo (np.array): the node features (batch * MAX_ATOM * N_FEAT).
        Ao (np.array): the edge pairs (batch * MAX_EDGE * 2).
        Eo (np.array): the edge features (batch * MAX_EDGE * E_FEAT).
        Mo (np.array): the mask of actual atoms (batch * MAX_ATOM).
        No (np.array): the inverse sqrt of node degrees for GCN attention 1/sqrt(N(i)*N(j)) (batch * MAX_EDGE).
    """

    # The adjacency matrix A, the first 6 elements of E are bond information.
    dim = int(E.shape[0]**0.5)
    E = np.reshape(E, (dim, dim, E.shape[-1]))

    A = E[..., :6].sum(axis=-1) != 0
    A = A.astype(np.float32)

    # The node feature Xo
    Xo = np.zeros(shape=(MAX_ATOM, N_FEAT))
    Xo[:X.shape[0], :X.shape[1]] = X

    # Convert A to edge pair format (if I use A_0 = np.zeros(...), the 0 to 0 pair will be emphasized a lot)
    # So I set all to the max_atom, then the max_atom atom has no node features.
    # And I mask all calculation for existing atoms.
    Ao = np.ones(shape=(MAX_EDGE + MAX_ATOM, 2)) * (MAX_ATOM - 1)
    A = sp.coo_matrix(A)
    n_edge = len(A.row)
    Ao[:n_edge, 0] = A.row
    Ao[:n_edge, 1] = A.col

    # The edge feature Eo
    Eo = np.zeros(shape=(MAX_EDGE + MAX_ATOM, E_FEAT))

    Eo[:n_edge, :] = [e[a.row, a.col] for e, a in zip([E], [A])][0]

    # Fill the zeros in Ao with self loop
    Ao[MAX_EDGE:, 0] = np.array([i for i in range(MAX_ATOM)])
    Ao[MAX_EDGE:, 1] = np.array([i for i in range(MAX_ATOM)])

    # The mask for existing nodes
    Mo = np.zeros(shape=(MAX_ATOM,))
    Mo[:X.shape[0]] = 1

    # The inverse of sqrt of node degrees
    outputa = np.unique(Ao[:, 0], return_counts=True, return_inverse=True)
    outputb = np.unique(Ao[:, 1], return_counts=True, return_inverse=True)
    n_a = []
    for element in outputa[1]:
        n_a.append(outputa[2][element])
    n_b = []
    for element in outputb[1]:
        n_b.append(outputb[2][element])
    n_a = np.array(n_a)
    n_b = np.array(n_b)
    no = np.multiply(n_a, n_b)
    No = 1 / np.sqrt(no)

    return Xo, Ao, Eo, Mo, No, n_edge


def load_molnet_data(func, featurizer, split, seed, MAX_ATOM, MAX_EDGE, N_FEAT, E_FEAT):
    """Load data from molecule-net datasets.
    Check https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function for details.
    Args:
        func (function): data loading function from molecule-net.
        featurizer (string): by default, Weave featurizer is used.
        split (string): the method to split data, e.g. stratified.
        seed (int): the random seed used to split data.
        MAX_ATOM (int): the maximum number of atoms zero-padding to.
        MAX_EDGE (int): the maximum number of edges zero-padding to.
        N_FEAT (int): the number of node features.
        E_FEAT (int): the number of edge features.

    Returns:
        X_train (list): training data.
        y_train (np.array): training labels.
        X_valid (list): validation data.
        y_valid (np.array): validation labels.
        X_test (list): testing data.
        y_test (np.array): testing labels.
        tasks (string): task name.
        transformers (class): contains functions to normalize and denormalize data.
    """
    tasks, \
    (train_dataset, valid_dataset, test_dataset), \
    transformers = func(featurizer=featurizer,
                        splitter=split,
                        seed=seed)

    X_train, X_valid, X_test = [], [], []
    A_train, A_valid, A_test = [], [], []
    E_train, E_valid, E_test = [], [], []
    y_train, y_valid, y_test = [], [], []
    M_train, M_valid, M_test = [], [], []
    N_train, N_valid, N_test = [], [], []
    train_x = train_dataset.X
    train_y = train_dataset.y
    valid_x = valid_dataset.X
    valid_y = valid_dataset.y
    test_x = test_dataset.X
    test_y = test_dataset.y

    # TRAINING DATASET
    max_atom = 0
    max_edge = 0
    for i in range(len(train_dataset)):
        try:
            X, A, E, M, N, n_edge = convert_data(train_x[i].nodes, train_x[i].pairs, MAX_ATOM, MAX_EDGE, N_FEAT, E_FEAT)
        except:
            continue
        X_train.append(X)
        E_train.append(E)
        A_train.append(A)
        M_train.append(M)
        N_train.append(N)
        y_train.append(train_y[i])
        if train_x[i].num_atoms > max_atom:
            max_atom = train_x[i].num_atoms
        if n_edge > max_edge:
            max_edge = n_edge
        print(f"max atom: {max_atom}, max edge: {max_edge}", end="\r")



    # VALIDATION DATASET
    for i in range(len(valid_dataset)):
        X, A, E, M, N, n_edge = convert_data(valid_x[i].nodes, valid_x[i].pairs, MAX_ATOM, MAX_EDGE, N_FEAT, E_FEAT)
        X_valid.append(X)
        E_valid.append(E)
        A_valid.append(A)
        M_valid.append(M)
        N_valid.append(N)
        y_valid.append(valid_y[i])
        if valid_x[i].num_atoms > max_atom:
            max_atom = valid_x[i].num_atoms
        if n_edge > max_edge:
            max_edge = n_edge
        print(f"max atom: {max_atom}, max edge: {max_edge}", end="\r")

    # TESTING DATASET
    for i in range(len(test_dataset)):
        X, A, E, M, N, n_edge = convert_data(test_x[i].nodes, test_x[i].pairs, MAX_ATOM, MAX_EDGE, N_FEAT, E_FEAT)
        X_test.append(X)
        E_test.append(E)
        A_test.append(A)
        M_test.append(M)
        N_test.append(N)
        y_test.append(test_y[i])
        if test_x[i].num_atoms > max_atom:
            max_atom = test_x[i].num_atoms
        if n_edge > max_edge:
            max_edge = n_edge
        print(f"max atom: {max_atom}, max edge: {max_edge}", end="\r")

    X_train = np.array(X_train)
    A_train = np.array(A_train)
    E_train = np.array(E_train)
    M_train = np.array(M_train)
    N_train = np.array(N_train)
    y_train = np.array(y_train).squeeze()
    X_valid = np.array(X_valid)
    A_valid = np.array(A_valid)
    E_valid = np.array(E_valid)
    M_valid = np.array(M_valid)
    N_valid = np.array(N_valid)
    y_valid = np.array(y_valid).squeeze()
    X_test = np.array(X_test)
    A_test = np.array(A_test)
    E_test = np.array(E_test)
    M_test = np.array(M_test)
    N_test = np.array(N_test)
    y_test = np.array(y_test).squeeze()

    X_train = [X_train, A_train, E_train, M_train, N_train]
    X_valid = [X_valid, A_valid, E_valid, M_valid, N_valid]
    X_test = [X_test, A_test, E_test, M_test, N_test]

    return X_train, y_train, X_valid, y_valid, X_test, y_test, tasks, transformers
