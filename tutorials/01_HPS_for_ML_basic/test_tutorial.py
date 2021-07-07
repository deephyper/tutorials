import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.join(HERE, "dh_project")
PROBLEM_PATH = os.path.join(PROJECT_PATH, "dh_project", "rf_tuning")

sys.path.insert(0, PROJECT_PATH)

def execute(script_file):
    script = os.path.join(PROBLEM_PATH, script_file)
    retval = os.system(f"{sys.executable} {script}")
    return retval

def test_baseline():
    retval = execute("run_baseline.py")
    assert retval == 0

def test_load_data():
    retval = execute("load_data.py")
    assert retval == 0

def test_problem():
    retval = execute("problem.py")
    assert retval == 0

def test_command():
    python_path = ":".join(sys.path)
    retval = os.system(f"PYTHONPATH={python_path} deephyper hps ambs --problem dh_project.rf_tuning.problem.Problem --run dh_project.rf_tuning.model_run.run --max-evals 2 --evaluator subprocess --n-jobs 4")
    assert retval == 0