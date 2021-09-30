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
    module load conda/2021-09-22

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
    INIT_SCRIPT=$PWD/init-dh-environment.sh
    CPUS_PER_NODE=8
    GPUS_PER_NODE=8

    # Initialize environment
    source $INIT_SCRIPT

    # Getting the node names
    mapfile -t nodes_array -d '\n' < $COBALT_NODEFILE

    head_node=${nodes_array[0]}
    head_node_ip=$(dig $head_node a +short | awk 'FNR==2')

    # if we detect a space character in the head node IP, we'll
    # convert it to an ipv4 address. This step is optional.
    if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
    head_node_ip=${ADDR[1]}
    else
    head_node_ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
    fi

    # Starting the Ray Head Node
    port=6379
    ip_head=$head_node_ip:$port
    export ip_head
    echo "IP Head: $ip_head"

    echo "Starting HEAD at $head_node"
    ssh -tt $head_node_ip "source $INIT_SCRIPT; cd $EXP_DIR; \
        ray start --head --node-ip-address=$head_node_ip --port=$port \
        --num-cpus $CPUS_PER_NODE --num-gpus $GPUS_PER_NODE --block" &

    # optional, though may be useful in certain versions of Ray < 1.0.
    sleep 10

    # number of nodes other than the head node
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

    # Execute the DeepHyper Task
    # Here the task is an hyperparameter search using the DeepHyper CLI
    # However it is also possible to call a Python script using different
    # Features from DeepHyper (see following notes)
    ssh -tt $head_node_ip "source $INIT_SCRIPT && cd $EXP_DIR && \
        deephyper hps ambs \
        --problem deephyper.benchmark.nas.linearRegHybrid.Problem \
        --evaluator ray \
        --run-function deephyper.nas.run.quick.run \
        --ray-address auto \
        --ray-num-cpus-per-task 1"

    # Stop de Ray cluster
    ssh -tt $head_node_ip "source $INIT_SCRIPT && ray stop"