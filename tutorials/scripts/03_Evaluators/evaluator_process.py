from deephyper.evaluator import ProcessPoolEvaluator
from ackley import run
from common import NUM_WORKERS, execute_search, plot_sum_up

SEARCH_TIMEOUT = 20

process_evaluator = ProcessPoolEvaluator(
    run,
    num_workers=NUM_WORKERS,
)

results, init_duration = execute_search(process_evaluator, SEARCH_TIMEOUT)
results.to_csv("results.csv")

plot_sum_up("process_evaluator")