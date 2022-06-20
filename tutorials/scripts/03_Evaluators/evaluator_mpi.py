from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from deephyper.evaluator import Evaluator
from ackley import run
from common import evaluate_and_plot

SEARCH_TIMEOUT = 20

with Evaluator.create(
    run,
    method='mpicomm',
) as evaluator:
    if evaluator is not None :
        evaluate_and_plot(evaluator, SEARCH_TIMEOUT, "mpi_evaluator")