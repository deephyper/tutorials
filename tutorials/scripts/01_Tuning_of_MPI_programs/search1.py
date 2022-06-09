import os
import pathlib
import subprocess

from search0 import _parse_results


class Experiment:
    """A simple context manager to do some initialization and finalization before run."""

    def __init__(self, config):
        self.config = config
        self.prev_dir = None
        self.exp_dir = None

    # Any initialization, data preparation, configuration can be done on enter
    def __enter__(self):
        self.prev_dir = os.getcwd()

        self.exp_dir = os.path.abspath(f"exp-{self.config['job_id']}")
        pathlib.Path(self.exp_dir).mkdir(parents=False, exist_ok=False)
        os.chdir(self.exp_dir)

        return self.exp_dir

    def __exit__(self, type, value, traceback):
        os.chdir(self.prev_dir)


def run_mpi(config, dequed=['localhost:1']):
    exe = os.path.dirname(os.path.abspath(__file__)) + "/f_exe"
    nodes = dequed
    runner = [
        'mpirun',
        ' -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH'
        ' --host ' + ','.join(nodes) + ' ',
        exe,
        f' {config["arg"]}'
    ]

    # context manager creates a dirrectory for current run
    # and moves current execution dirrectory there
    # when exit context manager changes current dirrectory bask
    with Experiment(config) as exp_dir:
        with open('stdout.txt', 'wb') as out, open('stderr.txt', 'wb') as err:
            subprocess.run(
                runner,
                stdout=out,
                stderr=err,
            )
        with open('runcommand.txt', 'w') as f:
            f.write(''.join(runner))

        # We can parse any results here
        with open(os.path.join(exp_dir, 'stdout.txt'), 'r') as f:
            res = _parse_results(f.read())

    return res


def test_run_mpi():
    config = {"job_id": 0, "arg": 2}
    result = run_mpi(config, dequed=['localhost:1'])
    print("result: ", result)


if __name__ == "__main__":
    test_run_mpi()
