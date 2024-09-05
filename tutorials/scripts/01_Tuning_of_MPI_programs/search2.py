from functools import partial

from deephyper.hpo import HpProblem
from deephyper.hpo import CBO
from deephyper.evaluator import SubprocessEvaluator

from search1 import run_mpi


def execute_deephyper():
    import logging
    logging.basicConfig(level=10)
    problem = HpProblem()
    problem.add_hyperparameter((-40., 40.), "arg")

    evaluator = SubprocessEvaluator(
        run_mpi, num_workers=1,
    )

    search = CBO(problem, evaluator)
    results = search.search(max_evals=10)
    print(results)


if __name__ == "__main__":
    execute_deephyper()
