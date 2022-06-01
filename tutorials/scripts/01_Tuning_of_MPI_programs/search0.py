import os
import parse
import subprocess


def _parse_results(stdout):
    res = parse.search('f(x) = {:f}', stdout)
    return res[0]


def run_mpi(config):
    exe = os.path.abspath(f"./f_exe")

    result = subprocess.run(
        [
            'mpirun',
            ' -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH'
            ' --host localhost:1 ',
            exe,
            f' {config["arg"]}'
        ],
        stdout=subprocess.PIPE,
    )

    res = _parse_results(result.stdout.decode("utf-8"))

    return res


def test_run_mpi():
    config = {"nodes_per_task": 2, "arg": -2}
    result = run_mpi(config)
    print("result: ", result)


if __name__ == "__main__":
    test_run_mpi()
