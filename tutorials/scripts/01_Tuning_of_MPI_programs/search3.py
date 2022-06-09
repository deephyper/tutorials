import os
import logging

from deephyper.evaluator.callback import LoggerCallback
from deephyper.problem import HpProblem
from deephyper.search.hps import CBO
from deephyper.evaluator import SubprocessEvaluator, queued


from search1 import run_mpi


def get_thetagpu_nodelist():
    f = os.environ['COBALT_NODEFILE']
    with open(f) as f:
        lines = f.readlines()
        nodelist = [line.rstrip() for line in lines]
    return nodelist


def execute_deephyper():
    logging.basicConfig(level=10)

    problem = HpProblem()
    problem.add_hyperparameter((-40., 40.), "arg")

    nodes_per_task = 2
    n_ranks_per_node = 8
    nodes = [f'{n}:{n_ranks_per_node}' for n in get_thetagpu_nodelist()]

    evaluator = queued(SubprocessEvaluator)(
        run_mpi,
        num_workers=len(nodes) // nodes_per_task,
        queue=nodes,
        queue_pop_per_task=nodes_per_task,
        callbacks=[LoggerCallback()]
    )

    search = CBO(problem, evaluator)
    results = search.search(max_evals=40)
    print(results)


if __name__ == "__main__":
    execute_deephyper()
