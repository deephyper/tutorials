from deephyper.problem import NaProblem
from dh_project.multi_input_search.load_data import load_data
from dh_project.multi_input_search.search_space import create_search_space

Problem = NaProblem(seed=2019)

Problem.load_data(load_data)

Problem.search_space(create_search_space, num_layers=3)

Problem.hyperparameters(
    batch_size=64,
    learning_rate=0.001,
    optimizer="adam",
    num_epochs=200,
    callbacks=dict(
        EarlyStopping=dict(
            monitor="val_r2", mode="max", verbose=0, patience=5  # or 'val_acc' ?
        ),
        ModelCheckpoint=dict(
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
            filepath="model.h5",
            save_weights_only=False,
        ),
    ),
)

Problem.loss("mse")

Problem.metrics(["r2"])

Problem.objective("val_r2")

# Just to print your problem, to test its definition and imports in the current python environment.
if __name__ == "__main__":
    print(Problem)