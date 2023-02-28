from mpi4py import MPI

from deephyper.search.hps import MPIDistributedBO


from ackley import hp_problem, run

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Each rank creates a RedisStorage client and connects to the storage server
# indicated by host:port. Then, the storage is passed to the evaluator.
evaluator = MPIDistributedBO.bootstrap_evaluator(
    run,
    evaluator_type="serial",
    storage_type="redis",
    storage_kwargs={
        "host": "localhost",
        "port": 6379,
    },
    comm=comm,
    root=0,
)

# A new search was created by the bootstrap_evaluator function.
if rank == 0:
    print(f"Search Id: {evaluator._search_id}")

# The Distributed Bayesian Optimization search instance is created
# With the corresponding evaluator and communicator.
search = MPIDistributedBO(
    hp_problem, evaluator, log_dir="mpi-distributed-log", comm=comm
)

# The search is started with a timeout of 10 seconds.
results = search.search(timeout=10)
