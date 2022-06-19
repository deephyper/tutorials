from deephyper.evaluator import ThreadPoolEvaluator
from ackley import run
from common import NUM_WORKERS, execute_search, plot_sum_up

SEARCH_TIMEOUT = 120

thread_evaluator = ThreadPoolEvaluator(
    run,
    num_workers=NUM_WORKERS,
)

results, init_duration = execute_search(thread_evaluator, SEARCH_TIMEOUT)
results.to_csv("results.csv")

plot_sum_up("thread_evaluator")