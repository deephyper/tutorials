import os
import pathlib
import subprocess
import datetime


class Experiment:
    """A simple context manager to move in/out of an experiment dictionnary."""

    def __init__(self, config):
        self.binary = '/home/boyda/proj/MG/trial-dh0.3/src/a.out'
        self.general_args = ''
        self.exp_args = None
        self.config = config
        self.prev_dir = None
        self.exp_dir = None

    def _prepare_args(self):
        self.exp_args = f"arg_name {self.config['arg']}"

    def __enter__(self):
        self._prepare_args()

        self.prev_dir = os.getcwd()

        exp_dir = "../dh-base/" \
            + f"/exp-{self.config['id']}"
        self.exp_dir = os.path.abspath(exp_dir)
        pathlib.Path(self.exp_dir).mkdir(parents=True, exist_ok=False)
        os.chdir(self.exp_dir)

        return self

    def __exit__(self, type, value, traceback):
        os.chdir(self.prev_dir)

    def parse_results(self, stdout):
        """A example of how to called results from a subprocess execution."""
        res = 0
        for line in stdout:
            nums = [int(s) for s in stdout.split() if s.isdigit()]
            res += nums[0]
        return res


def test_mpi_bin():
    from mpi4py import MPI
    # import sys

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()

    with open('stdout.txt', 'w') as f:
        f.write(f"Hello, World! I am process {rank} of {size} on {name}.\n")


def run_mpi(config):
    with Experiment(config) as exp:
        test_mpi_bin()

        # result = subprocess.run(
        #     [
        #         exp.binary,
        #         exp.general_args,
        #         exp.exp_args,
        #     ],
        #     stdout=subprocess.PIPE,
        # )
        # with open('stdout.txt', 'w') as f:
        #     f.write(result.stdout.decode("utf-8"))
        # result = exp.parse_results(result.stdout.decode("utf-8"))

    return result


def execute_deephyper():

    from deephyper.evaluator import RayEvaluator
    from deephyper.evaluator.callback import LoggerCallback
    from deephyper.hpo import HpProblem
    from deephyper.hpo import CBO

    problem = HpProblem()
    problem.add_hyperparameter((0, 16), "arg")

    print("PRERAPTION ------------------------")

    evaluator = RayEvaluator(
        run_mpi, address='auto', num_cpus_per_task=4, num_gpus_per_task=4,
        callbacks=[LoggerCallback()],
    )
    print("Num Ray workers:", evaluator.num_workers)

    search = CBO(problem, evaluator)

    results = search.search(max_evals=10)

    print(results)


if __name__ == "__main__":
    execute_deephyper()

