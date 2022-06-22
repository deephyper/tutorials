from deephyper.search.hps import DBO

from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

from ackley import hp_problem, run


search = DBO(
    hp_problem,
    run,
    sync_communication=False,
    log_dir=".",
    checkpoint_file="results.csv",
    checkpoint_freq=1,
)

timeout = 10
if rank == 0:
    results = search.search(timeout=timeout)
    results.to_csv("results.csv")
else:
    search.search(timeout=timeout)
