from deephyper.evaluator import Evaluator
from ackley import run
from common import NUM_WORKERS, evaluate_and_plot

SEARCH_TIMEOUT = 20

evaluator = Evaluator.create(
    run,
    method='thread',
    method_kwargs=dict(
        num_workers=NUM_WORKERS,
    )
)

evaluate_and_plot(evaluator, SEARCH_TIMEOUT, "thread_evaluator")