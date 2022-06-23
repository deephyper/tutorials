if __name__ == "__main__":
    from deephyper.evaluator import Evaluator
    from ackley import run
    from common import evaluate_and_plot

    with Evaluator.create(
        run,
        method="mpicomm",
    ) as evaluator:
        if evaluator is not None:
            evaluate_and_plot(evaluator, "mpi_evaluator")
