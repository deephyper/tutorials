.. _tutorial-alcf-02:

Execution on the ThetaGPU supercomputer (with Ray)
**************************************************

In this tutorial we are going to learn how to use DeepHyper on the **ThetaGPU** supercomputer at the ALCF using Ray. `ThetaGPU <https://www.alcf.anl.gov/support-center/theta/theta-thetagpu-overview>`_ is a 3.9 petaflops system based on NVIDIA DGX A100.

Submission Script
=================

This section of the tutorial will show you how to submit script to the COBALT scheduler of ThetaGPU. To execute DeepHyper on ThetaGPU with a submission script it is required to:

1. Define a Bash script to initialize the environment (e.g., load a module, activate a conda environment).
2. Define a script composed of 3 steps: (1) launch a Ray cluster on available ressources, (2) execute a Python application which connects to the Ray cluster, and (3) stop the Ray cluster.

Start by creating a script named ``init-dh-environment.sh`` to initialize your environment. It will be used to initialize each used compute node. Replace the ``$CONDA_ENV_PATH`` by your personnal conda installation (e.g., it can be replaced by ``base`` if no virtual environment is used):


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

Add the following content:

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
    CPUS_PER_NODE=8
    GPUS_PER_NODE=8

    # Initialization of environment
    source $INIT_SCRIPT

    # Getting the node names
    mapfile -t nodes_array -d '\n' < $COBALT_NODEFILE

    head_node=${nodes_array[0]}
    head_node_ip=$(dig $head_node a +short | awk 'FNR==2')

    # Starting the Ray Head Node
    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    echo "Starting HEAD at $head_node"
    ssh -tt $head_node_ip "source $INIT_SCRIPT; cd $EXP_DIR; \
        ray start --head --node-ip-address=$head_node_ip --port=$port \
        --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE --block" &

    # Optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10

    # Number of nodes other than the head node
    worker_num=$((${#nodes_array[*]} - 1))
    echo "$worker_num workers"

    for ((i = 1; i <= worker_num; i++)); do
        node_i=${nodes_array[$i]}
        node_i_ip=$(dig $node_i a +short | awk 'FNR==1')
        echo "Starting WORKER $i at $node_i with ip=$node_i_ip"
        ssh -tt $node_i_ip "source $INIT_SCRIPT; cd $EXP_DIR; \
            ray start --address $ip_head \
            --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE --block" &
        sleep 5
    done

    # Check the status of the Ray cluster
    ssh -tt $head_node_ip "source $INIT_SCRIPT && ray status"

    # Run the search
    ssh -tt $head_node_ip "source $INIT_SCRIPT && cd $EXP_DIR && python myscript.py"

    # Stop de Ray cluster
    ssh -tt $head_node_ip "source $INIT_SCRIPT && ray stop"

.. note::

    .. code-block:: bash

        #COBALT --attrs filesystems=home,grand,eagle,theta-fs0
    
    The ``filesystems`` attribute corresponds to the filesystems your application should have access to, DeepHyper only requires ``home`` and ``theta-fs0``, and it is unnecessary to let in this list a filesystem your application (and DeepHyper) doesn't need.

Adapt the executed Python application depending on your needs. Here is an application axample of ``CBO`` using the ``ray`` evaluator:

.. code-block:: python
    :caption: **file**: ``myscript.py``

    import pathlib
    import os

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    import numpy as np

    from deephyper.evaluator import Evaluator
    from deephyper.search.hps import CBO
    from deephyper.evaluator.callback import ProfilingCallback

    from deephyper.problem import HpProblem


    hp_problem = HpProblem()
    hp_problem.add_hyperparameter((-10.0, 10.0), "x")

    def run(config):
        return - config["x"]**2

    timeout = 10
    search_log_dir = "results/cbo/"
    pathlib.Path(search_log_dir).mkdir(parents=False, exist_ok=True)

    # Evaluator creation
    print("Creation of the Evaluator...")
    evaluator = Evaluator.create(
        run,
        method="ray",
        method_kwargs={
            "adress": "auto",
            "num_gpus_per_task": 1,
        }
    )
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

Finally, submit the script using:

.. code-block:: bash

    qsub-gpu deephyper-job.qsub

.. note::

    The ``ssh -tt $head_node_ip "source $INIT_SCRIPT && ray status"`` command is used to check the good initialization of the Ray cluster. Once the job starts running, check the ``*.output`` file and verify that the number of detected GPUs is correct.
