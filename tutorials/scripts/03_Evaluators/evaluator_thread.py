if __name__ == "__main__":
    from deephyper.evaluator import Evaluator
    from ackley import run
    from common import NUM_WORKERS, evaluate_and_plot

    wait_function = 'cpu_bound' #IO_bound

    evaluator = Evaluator.create(
        run,
        method='thread',
        method_kwargs=dict(
            num_workers=NUM_WORKERS,
            run_function_kwargs=dict(
                wait_function=wait_function
            )
        )
    )

    evaluate_and_plot(evaluator, f"thread_evaluator_{wait_function}")