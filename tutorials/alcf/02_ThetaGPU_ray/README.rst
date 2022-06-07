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

Edit the ``#COBALT ...`` directives:

.. code-block:: bash

    #COBALT -A $PROJECT_NAME
    #COBALT -n 2
    #COBALT -q full-node
    #COBALT -t 20
    #COBALT --attrs filesystems=home,grand,eagle,theta-fs0

and adapt the executed Python application depending on your needs:

.. code-block:: python

    myscript.py

Finally, submit the script from a ThetaGPU login node (e.g., ``thetagpusn1``):

.. code-block:: bash

    qsub deephyper-job.qsub

.. note::

    The ``ssh -tt $head_node_ip "source $INIT_SCRIPT && ray status"`` command is used to check the good initialization of the Ray cluster. Once the job starts running, check the ``*.output`` file and verify that the number of detected GPUs is correct.

Jupyter Notebook
================

This section of the tutorial will show you how to run an interactive Jupyter notebook on ThetaGPU. After logging in Theta:

1. From a ``thetalogin`` node: ``ssh thetagpusn1`` to login to a ThetaGPU service node.
2. From ``thetagpusn1``, start an interactive job (**note** which ``thetagpuXX`` node you get placed onto will vary) by replacing your ``$PROJECT_NAME`` and ``$QUEUE_NAME`` (e.g. of available queues are ``full-node`` and ``single-gpu``):

.. code-block:: console

    (thetagpusn1) $ qsub -I -A $PROJECT_NAME -n 1 -q $QUEUE_NAME -t 60
    Job routed to queue "full-node".
    Wait for job 10003623 to start...
    Opening interactive session to thetagpu21

3. Wait for the interactive session to start. Then, from the ThetaGPU compute node (`thetagpuXX`), execute the following commands to initialize your DeepHyper environment (adapt to your needs):

.. code-block:: console

    $ . /etc/profile
    $ module load conda/2021-09-22
    $ conda activate $CONDA_ENV_PATH

4. Then, start the Jupyter notebook server:

.. code-block:: console

    $ jupyter notebook &

.. note::

    In the case of a multi-GPUs node, it is possible that the Jupyter notebook process will lock one of the available GPUs. Therefore, launch the notebook with the following command instead:

    .. code-block:: console

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 jupyter notebook &

4. Take note of the hostname of the current compute node (e.g. ``thetagpuXX``):

.. code-block:: console

    echo $HOSTNAME

5. Leave the interactive session running and open a new terminal window on your local machine.

6. In the new terminal window, execute the SSH command to link the local port to the ThetaGPU compute node after replacing with you ``$USERNAME`` and corresponding ``thetagpuXX``:

.. code-block:: console

    $ ssh -tt -L 8888:localhost:8888 $USERNAME@theta.alcf.anl.gov "ssh -L 8888:localhost:8888 thetagpuXX"

7. Open the Jupyter URL (`http:localhost:8888/?token=....`) in a local browser. This URL was printed out at step 4.
