from deephyper.hpo import HpProblem

Problem = HpProblem()

Problem.add_hyperparameter((1e-4, 1e-1, "log-uniform"), "lr")


if __name__ == "__main__":
    print(Problem)
