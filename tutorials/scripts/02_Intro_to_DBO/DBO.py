import logging

from deephyper.search.hps import DBO

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from ackley import hp_problem, run


search = DBO(
    hp_problem,
    run,
)

timeout = 10
if rank == 0:
    results = search.search(timeout=timeout)
else:
    search.search(timeout=timeout)

if rank == 0:
    results.to_csv("results.csv")
