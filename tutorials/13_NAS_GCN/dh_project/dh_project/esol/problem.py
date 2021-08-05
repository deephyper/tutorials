from deephyper.problem import NaProblem
from dh_project.esol.load_data import load_data
from dh_project.esol.search_space import create_search_space

Problem = NaProblem(seed=2020)
Problem.load_data(load_data)
Problem.search_space(create_search_space, data='esol')
Problem.hyperparameters(
    batch_size=64,
    learning_rate=1e-3,
    optimizer='adam',
    num_epochs=80)
Problem.loss("mse")
Problem.metrics(['mae', 'mse', 'r2', 'negmse'])
Problem.objective('val_negmse__max')

if __name__ == '__main__':
    print(Problem)
