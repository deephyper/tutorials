if __name__ == "__main__":
    from deephyper.evaluator import Evaluator
    from ackley import run
    from common import NUM_WORKERS, evaluate_and_plot

    evaluator = Evaluator.create(
        run,
        method='subprocess',
        method_kwargs=dict(
            num_workers=NUM_WORKERS,
        )
    )

    evaluate_and_plot(evaluator, "subprocess_evaluator")