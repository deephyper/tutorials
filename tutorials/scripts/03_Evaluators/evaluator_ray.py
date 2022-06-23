if __name__ == "__main__":
    from deephyper.evaluator import Evaluator
    from ackley import run
    from common import NUM_WORKERS, evaluate_and_plot

    evaluator = Evaluator.create(
        run,
        method='ray',
        method_kwargs=dict(
            num_workers=NUM_WORKERS,
        )
    )

    evaluate_and_plot(evaluator, "ray_evaluator")