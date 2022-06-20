from deephyper.evaluator import Evaluator
from ackley import run
from common import NUM_WORKERS, execute_search, plot_sum_up

SEARCH_TIMEOUT = 20

evaluator = Evaluator.create(
    run,
    method='serial',
    method_kwargs=dict(
        num_workers=NUM_WORKERS,
    )
)

init_duration = execute_search(evaluator, SEARCH_TIMEOUT)

plot_sum_up("serial_evaluator")