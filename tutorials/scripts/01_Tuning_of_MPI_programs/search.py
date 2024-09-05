import re
import subprocess

from deephyper.evaluator import SubprocessEvaluator


class OneNodeEvaluator(SubprocessEvaluator):
    """A custom evaluator to manage node ressources and MPI calls in the run-function.
    It used num_workers parallel workers per task, where one task is execution of MPI
    function using nodes_per_task  nodes (CPUs or GPUs). 
    Total number of available nodes (CPUs or GPUs) equal to nodes_per_task * num_workers.

    Args:
        run_function (callable): the remote function to execute.
        num_workers (int, optional): the number of parallel search processes launched.
        nodes_per_task (int, optional): the number of nodes (CPUs or GPUs) per task used.
        callbacks (list, optional): A list of ``evaluator.callback``.
    """

    def __init__(
        self, run_function, num_workers=1, nodes_per_task=1, callbacks=None
    ):
        super().__init__(run_function, num_workers=num_workers, callbacks=callbacks)
        self.nodes_per_task = nodes_per_task

    async def execute(self, job):
        """Extend the execute function to add the 'nodes_per_task' in the configuration dictionnary."""
        job.config["nodes_per_task"] = self.nodes_per_task
        sol = await super().execute(job)
        job.config.pop("nodes_per_task")
        return sol


def _parse_results(stdout):
    """A example of how to called results from a subprocess execution."""
    g = re.findall("My arg is [0-9]", stdout)
    res = sum([int(s.split()[-1]) for s in g])
    return res


def run_mpi(config):

    nodes_per_task = config.get("nodes_per_task", 1)

    result = subprocess.run(
        [
            "mpirun",
            "-n",
            str(nodes_per_task),
            "echo",
            "Hello from DeepHyper! My arg is {}".format(config["arg"]),
            "&& sleep 30"
        ],
        stdout=subprocess.PIPE,
    )

    res = _parse_results(result.stdout.decode("utf-8"))

    return res


def test_run_mpi():
    config = {"id": 0, "nodes_per_task": 2, "arg": 2}
    result = run_mpi(config)
    print("result: ", result)


def execute_deephyper():

    from deephyper.evaluator.callback import LoggerCallback
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO

    problem = HpProblem()
    problem.add_hyperparameter((0, 16), "arg")

    evaluator = OneNodeEvaluator(
        run_mpi, num_workers=8, callbacks=[LoggerCallback()], nodes_per_task=2
    )

    search = CBO(problem, evaluator)

    results = search.search(max_evals=10)

    print(results)


if __name__ == "__main__":
    # test_run_mpi()
    execute_deephyper()

# Custom MPI code
# * preparation stage, prepare some input data and/or configuration for our application
# * run our application with custom environment, from custom folder
# * parse output of mpi code

# ! MPI code requires no more than one node

# ! MPI code requires several nodes
