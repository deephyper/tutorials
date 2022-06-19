from deephyper.evaluator import SubprocessEvaluator
from ackley import run
from common import NUM_WORKERS, execute_search, plot_sum_up

SEARCH_TIMEOUT = 20

suprocess_evaluator = SubprocessEvaluator(
    run,
    num_workers=NUM_WORKERS,
)

results, init_duration = execute_search(suprocess_evaluator, SEARCH_TIMEOUT)
results.to_csv("results.csv")

plot_sum_up("suprocess_evaluator")