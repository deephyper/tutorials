import os
import pathlib
import re
import subprocess

from deephyper.evaluator import SubprocessEvaluator


class CustomEvaluator(SubprocessEvaluator):
    """A custom evaluator to manage node ressources and MPI calls in the run-function.

    Args:
        run_function (callable): the remote function to execute.
        num_workers (int, optional): the number of parallel processes launched. Defaults to 1.
        callbacks (list, optional): A list of ``evaluator.callback``. Defaults to None.
        nodes_per_task (int, optional): the number of nodes per remote task used. Defaults to 2.
    """

    def __init__(
        self, run_function, num_workers: int = 1, callbacks=None, nodes_per_task=2
    ):
        super().__init__(run_function, num_workers=num_workers, callbacks=callbacks)
        # this simlpe example use a constant of the number of nodes per task
        # it is also possible to manage a queue of available nodes and .pop/.push from it
        self.nodes_per_task = nodes_per_task

    async def execute(self, job):
        """Extend the execute function to add the 'nodes_per_task' in the configuration dictionnary."""
        job.config["nodes_per_task"] = 2
        sol = await super().execute(job)
        job.config.pop("nodes_per_task")
        return sol


class Experiment:
    """A simple context manager to move in/out of an experiment dictionnary."""

    def __init__(self, config):
        self.config = config
        self.prev_dir = None
        self.exp_dir = None

    def __enter__(self):
        # we can add here all the stuff required to prepare our experiment
        self.prev_dir = os.getcwd()

        self.exp_dir = os.path.abspath(f"exp-{self.config['id']}")
        pathlib.Path(self.exp_dir).mkdir(parents=False, exist_ok=False)
        os.chdir(self.exp_dir)

        return self.exp_dir

    def __exit__(self, type, value, traceback):
        os.chdir(self.prev_dir)


def _parse_results(stdout):
    """A example of how to called results from a subprocess execution."""
    g = re.findall("My arg is [0-9]", stdout)
    res = sum([int(s.split()[-1]) for s in g])
    return res


def run_mpi(config):

    nodes_per_task = config.get("nodes_per_task", 2)

    # "with" statement used to move to an "exp_dir" and move back to initial PWD
    with Experiment(config) as exp_dir:

        result = subprocess.run(
            [
                "mpirun",
                "-n",
                str(nodes_per_task),
                "echo",
                "Hello from DeepHyper! My arg is {}".format(config["arg"]),
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
    from deephyper.problem import HpProblem
    from deephyper.search.hps import AMBS

    problem = HpProblem()
    problem.add_hyperparameter((0, 9), "arg")

    evaluator = CustomEvaluator(
        run_mpi, num_workers=1, callbacks=[LoggerCallback()], nodes_per_task=2
    )

    search = AMBS(problem, evaluator)

    results = search.search(max_evals=10)

    print(results)


if __name__ == "__main__":
    # test_run_mpi()
    execute_deephyper()
