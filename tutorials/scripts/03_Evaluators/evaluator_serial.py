from deephyper.evaluator import SerialEvaluator
from ackley import run
from common import NUM_WORKERS, execute_search, plot_sum_up

SEARCH_TIMEOUT = 20

serial_evaluator = SerialEvaluator(
    run,
    num_workers=NUM_WORKERS,
)

results, init_duration = execute_search(serial_evaluator, SEARCH_TIMEOUT)
results.to_csv("results.csv")

plot_sum_up("serial_evaluator", SEARCH_TIMEOUT)