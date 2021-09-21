.. _tutorial-alcf-02:

.. important::

    **Tutorial under construction!**


.. warning::

    Be sure to work in a virtual environment where you can easily ``pip install`` new packages. This typically entails using either Anaconda, virtualenv, or Pipenv. If you followed the standard DeepHyper installation procedure you can simply activate the created Conda environment.

Execution on the ThetaGPU supercomputer
***************************************

In this tutorial we are going to learn how to use DeepHyper on the **ThetaGPU** supercomputer at the ALCF. `ThetaGPU <https://www.alcf.anl.gov/support-center/theta/theta-thetagpu-overview>`_ is a 3.9 petaflops system based on NVIDIA DGX A100. It is composed of:

Then create a script named ``init-dh-environment.sh``:


.. code-block:: bash
    :caption: **file**: ``init-dh-environment.sh``

    #!/bin/bash

    # Necessary for Bash shells
    . /etc/profile

    # Tensorflow optimized for A100 with CUDA 11
    module load conda/2021-06-28

    # Activate conda env
    conda activate $CONDA_ENV_PATH

.. tip::

    This ``init-dh-environment`` script can be very useful to tailor the execution's environment to your needs. Here are a few tips that can be useful:

    - To activate XLA optimized compilation add ``export TF_XLA_FLAGS=--tf_xla_enable_xla_devices``
    - To change the log level of Tensorflow add ``export TF_CPP_MIN_LOG_LEVEL=3``


.. code-block:: bash
    :caption: **file**: ``deephyper-job.qsub``

    #!/bin/bash +x

    # User Configuration
    EXP_DIR=$PWD
    INIT_SCRIPT=$PWD/../SetUpEnv.sh
    CPUS_PER_NODE=8
    GPUS_PER_NODE=8

    # Initialization of environment
    source $INIT_SCRIPT

    # Collect IP addresses of available compute nodes
    mapfile -t nodes_array -d '\n' < $COBALT_NODEFILE
    HEAD_NODE=${nodes_array[0]}
    HEAD_NODE_IP=$(dig $HEAD_NODE a +short | awk 'FNR==2')
    echo "Detected HEAD node $HEAD_NODE with IP $HEAD_NODE_IP"

    WORKER_NODES=${nodes_array[@]:1}
    for ((i=0; i < ${#WORKER_NODES[@]}; i++)); do
        WORKER_NODES_IPS[$i]=$(dig ${WORKER_NODES[$i]} a +short | awk 'FNR==1')
    done
    echo "Detected ${#WORKER_NODES[@]} workers with IPs: ${WORKER_NODES_IPS[@]}"

    # Launch the Ray cluster
    # Starting the Ray Head Node
    RAY_PORT=6379

    echo "Starting HEAD at $HEAD_NODE_IP"
    ssh -tt $HEAD_NODE_IP "source $INIT_SCRIPT; \
        ray start --head --node-ip-address=$HEAD_NODE_IP --port=$RAY_PORT \
        --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE" &

    # optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10

    # number of nodes other than the head node
    for ((i=0; i < ${#WORKER_NODES_IPS[@]}; i++)); do
        echo "Starting WORKER $i at ${WORKER_NODES_IPS[$i]}"
        ssh -tt ${WORKER_NODES_IPS[$i]} "source $INIT_SCRIPT && \
            ray start --address $HEAD_NODE_IP:$RAY_PORT \
            --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE" &
        sleep 5
    done

    # Execute the DeepHyper Task
    # Here the task is an hyperparameter search using the DeepHyper CLI
    # However it is also possible to call a Python script using different
    # Features from DeepHyper (see following notes)
    ssh $HEAD_NODE_IP "source $INIT_SCRIPT && cd $EXP_DIR && \
        deephyper hps ambs \
        --problem deephyper.benchmark.nas.linearRegHybrid.Problem \
        --evaluator ray \
        --run-function deephyper.nas.run.quick.run \
        --ray-address auto \
        --ray-num-cpus-per-task 1"

    # Stop de Ray cluster
    for ((i=0; i < ${#WORKER_NODES_IPS[@]}; i++)); do
        echo "Stopping WORKER $i at ${WORKER_NODES_IPS[$i]}"
        ssh -tt ${WORKER_NODES_IPS[$i]} "source $INIT_SCRIPT && ray stop"
        sleep 5
    done