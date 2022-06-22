import time
import numpy as np
from deephyper.problem import HpProblem

d = 10
domain = (-32.768, 32.768)
hp_problem = HpProblem()
for i in range(d):
    hp_problem.add_hyperparameter(domain, f"x{i}")

def ackley(x, a=20, b=0.2, c=2*np.pi):
    d = len(x)
    s1 = np.sum(x ** 2)
    s2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(s1 / d))
    term2 = -np.exp(s2 / d)
    y = term1 + term2 + a + np.exp(1)
    return y

def run(config):
    x = np.array([config[f"x{i}"] for i in range(d)])
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    return -ackley(x)