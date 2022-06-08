.. _tutorial-alcf-03:

Execution on the ThetaGPU supercomputer (with MPI)
**************************************************

In this tutorial we are going to learn how to use DeepHyper on the **ThetaGPU** supercomputer at the ALCF using MPI. `ThetaGPU <https://www.alcf.anl.gov/support-center/theta/theta-thetagpu-overview>`_ is a 3.9 petaflops system based on NVIDIA DGX A100.

Submission Script
=================

This section of the tutorial will show you how to submit script to the COBALT scheduler of ThetaGPU. To execute DeepHyper on ThetaGPU with a submission script it is required to:

1. Define a Bash script to initialize the environment (e.g., load a module, activate a conda environment).
2. Define an execution script, which will call the bash script defined in 1. and launch the python script using ``mpirun``.

Start by creating a script named ``init-dh-environment.sh`` to initialize your environment. Replace the ``$CONDA_ENV_PATH`` by your personnal conda installation (e.g., it can be replaced by ``base`` if no virtual environment is used):


.. code-block:: bash
    :caption: **file**: ``init-dh-environment.sh``

    #!/bin/bash

    # Necessary for Bash shells
    . /etc/profile

    # Tensorflow optimized for A100 with CUDA 11
    module load conda/2021-11-30

    # Activate conda env
    conda activate $CONDA_ENV_PATH

.. tip::

    This ``init-dh-environment`` script can be very useful to tailor the execution's environment to your needs. Here are a few tips that can be useful:

    - To activate XLA optimized compilation add ``export TF_XLA_FLAGS=--tf_xla_enable_xla_devices``
    - To change the log level of Tensorflow add ``export TF_CPP_MIN_LOG_LEVEL=3``


Then create a new file named ``deephyper-job.qsub`` and make it executable. It will correspond to your submission script.

.. code-block:: bash

    $ touch deephyper-job.qsub && chmod +x deephyper-job.qsub

Add the following content (adapt ``$PROJECT_NAME`` to your current project, e-g ``#COBALT -A datascience``):

.. code-block:: bash
    :caption: **file**: ``deephyper-job.qsub``

    #!/bin/bash
    #COBALT -A $PROJECT_NAME
    #COBALT -n 2
    #COBALT -q full-node
    #COBALT -t 20
    #COBALT --attrs filesystems=home,grand,eagle,theta-fs0

    # User Configuration
    EXP_DIR=$PWD
    INIT_SCRIPT=$PWD/init-dh-environment.sh
    COBALT_JOBSIZE=2
    RANKS_PER_NODE=8

    # Initialization of environment
    source $INIT_SCRIPT

    mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE --hostfile $COBALT_NODEFILE python myscript.py

.. note::

    .. code-block:: bash

        #COBALT --attrs filesystems=home,grand,eagle,theta-fs0
    
    The ``filesystems`` attribute corresponds to the filesystems your application should have access to, DeepHyper only requires ``home`` and ``theta-fs0``, and it is unnecessary to let in this list a filesystem your application (and DeepHyper) doesn't need.

.. note::

    .. code-block:: bash

        COBALT_JOBSIZE=2
        RANKS_PER_NODE=8

    ``COBALT_JOBSIZE`` and ``RANKS_PER_NODE`` correspond respectively to the number of nodes allocated and number of process per node. Unlike ``Theta`` on which ``COBALT_JOBSIZE`` is automatically instanciated to the correct value, on ``ThetaGPU`` it has to be done by hand : ``COBALT_JOBSIZE`` should always correspond to the number of nodes you submitted your application to (the number after ``#COBALT -n``). e-g if you were to happen to have a ``#COBALT -n 4`` you should have ``COBALT_JOBSIZE=4``.

Adapt the executed Python application depending on your needs. Here is an application axample of ``CBO`` using the ``mpi_comm`` evaluator:

.. code-block:: python
    :caption: **file**: ``myscript.py``

    import pathlib
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import mpi4py

    mpi4py.rc.initialize = False
    mpi4py.rc.threads = True
    mpi4py.rc.thread_level = "multiple"

    from deephyper.evaluator import Evaluator
    from deephyper.search.hps import CBO

    from mpi4py import MPI

    if not MPI.Is_initialized():
        MPI.Init_thread()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    from deephyper.problem import HpProblem


    hp_problem = HpProblem()
    hp_problem.add_hyperparameter((-10.0, 10.0), "x")

    def run(config):
        return - config["x"]**2

    timeout = 10
    search_log_dir = "results/cbo/"
    pathlib.Path(search_log_dir).mkdir(parents=False, exist_ok=True)

    if rank == 0:
        # Evaluator creation
        print("Creation of the Evaluator...")

    with Evaluator.create(
        run,
        method="mpicomm",
    ) as evaluator:
        if evaluator is not None:
            print(f"Creation of the Evaluator done with {evaluator.num_workers} worker(s)")

            # Search creation
            print("Creation of the search instance...")
            search = CBO(
                hp_problem,
                evaluator,
            )
            print("Creation of the search done")

            # Search execution
            print("Starting the search...")
            results = search.search(timeout=timeout)
            print("Search is done")

            results.to_csv(os.path.join(search_log_dir, f"results.csv"))

.. note::

    To ensure that each worker is restricted to its own gpu (and doesn't access other workers memory) you might need to add this to your script:

    .. code-block:: python

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        gpu_local_idx = rank % gpu_per_node
        node = int(rank / gpu_per_node)

        import tensorflow as tf
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.set_visible_devices(gpus[gpu_local_idx], "GPU")
                tf.config.experimental.set_memory_growth(gpus[gpu_local_idx], True)
                logical_gpus = tf.config.list_logical_devices("GPU")
                logging.info(f"[r={rank}]: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
            except RuntimeError as e:
                # Visible devices must be set before GPUs have been initialized
                logging.info(f"{e}") 
    
    With ``gpu_per_node`` being equal to the ``RANKS_PER_NODE`` specified in the submission script.

Finally, submit the script using :

.. code-block:: bash

    qsub-gpu deephyper-job.qsub
