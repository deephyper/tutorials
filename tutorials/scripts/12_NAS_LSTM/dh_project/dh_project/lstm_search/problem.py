from deephyper.problem import NaProblem
from dh_project.lstm_search.load_data import load_data
from dh_project.lstm_search.search_space import create_search_space

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.search_space(create_search_space, num_layers=5)

Problem.hyperparameters(
    batch_size=32,
    learning_rate=0.001,
    optimizer='adam',
    num_epochs=100,
    callbacks=dict(
        EarlyStopping=dict(
            monitor='val_r2',
            mode='max',
            verbose=0,
            patience=5
        )
    )
)

Problem.loss('mse')

Problem.metrics(['r2'])

Problem.objective('val_r2')


# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == '__main__':
    print(Problem)