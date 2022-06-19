from deephyper.evaluator import RayEvaluator
from ackley import run
from common import NUM_WORKERS, execute_search, plot_sum_up

SEARCH_TIMEOUT = 120

ray_evaluator = RayEvaluator(
    run,
    num_workers=NUM_WORKERS,
)

results, init_duration = execute_search(ray_evaluator, SEARCH_TIMEOUT)
results.to_csv("results.csv")

plot_sum_up("ray_evaluator")