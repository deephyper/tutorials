from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from deephyper.evaluator import MPICommEvaluator
from ackley import run
from common import execute_search, plot_sum_up

SEARCH_TIMEOUT = 20

mpi_evaluator = MPICommEvaluator(
    run,
)

results, init_duration = execute_search(mpi_evaluator, SEARCH_TIMEOUT)

if rank == 0:
    results.to_csv("results.csv")
    plot_sum_up("mpi_evaluator")