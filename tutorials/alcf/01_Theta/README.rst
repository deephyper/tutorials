.. _tutorial-alcf-01:

.. important::

    **Tutorial under construction!**


.. warning::

    Be sure to work in a virtual environment where you can easily ``pip install`` new packages. This typically entails using either Anaconda, virtualenv, or Pipenv. If you followed the standard DeepHyper installation procedure you can simply activate the created Conda environment.

.. admonition:: Storage/File Systems
    :class: dropdown, important

    It is important to run the following commands from the appropriate storage space because some features of DeepHyper can generate a consequante quantity of data such as model checkpointing. The storage spaces available at the ALCF are:

    - ``/lus/grand/projects/``
    - ``/lus/eagle/projects/``
    - ``/lus/theta-fs0/projects/``

    For more details refer to `ALCF Documentation <https://www.alcf.anl.gov/support-center/theta/theta-file-systems>`_.


Execution on the Theta supercomputer
************************************

In this tutorial we are going to learn how to use DeepHyper on the **Theta** supercomputer at the ALCF. `Theta <https://www.alcf.anl.gov/support-center/theta/theta-thetagpu-overview>`_ is  a Cray XC40, 11.7 petaflops system based on Intel® Xeon Phi processor. It is composed of:

1. Login nodes ``thetalogin*``
2. Head nodes ``thetamom*``
3. Compute nodes ``nid*``

When logging in **Theta** you are located on a **login node**. From this node you can set up your environment such as downloading your data or installing the software you will use for your experimentations. Then, you will setup an experiment using DeepHyper and finally submit an allocation request with the ``qsub`` command line to execute your code on the compute nodes. Once the allocation starts you will be located on a **head node** from which you can access compute nodes using MPI (e.g., with the ``aprun`` command line) or using SSH. However, using SSH requires a specific arguments to be present in your allocation request otherwise it will be blocked by the system.

When using DeepHyper, one can use two different strategies to distribute the computation of evaluations on the supercomputer:

1. **1 evaluation per node**: many evaluations can be launched in parallel but each of them only uses the ressources of at most one node (e.g., one neural network training per node).
2. **1 evaluation per multiple nodes**: many evaluations can be launched in parallel and each of them can use the ressources of multiple nodes.

Strategy 1 - 1 evaluation per node
==================================

.. todo::

    add example scripts (``run`` and ``problem``).

In this strategy, many evaluations can be launched in parallel but each of them only uses the ressources of at most one node. For example, you can train one neural network per node. In this case, we will (1) start by launching a Ray cluster accross all compute node, then we will (2) start the search process on one of them and send tasks to previously instanciated workers.

For this, it is required to define a shell script which will initialize the environment of each compute node. One way to initialize your environment could be to use the ``~/.bashrc`` or ``~/.bash_profile`` which are called at the beginning of each session. However, if you want to have different installations depending on your experimentations it is preferable to avoid activating globally each installation but instead activate them only when necessary. To that end, we will create the ``init-dh-environment.sh`` script which will be called to initialize each compute node:

.. code-block:: console

    touch init-dh-environment.sh
    chmod +x init-dh-environment.sh

Once created and executable you can add the following content in it (e.g. ``vim init-dh-environment.sh``):

.. code-block:: bash
    :caption: **file**: ``init-dh-environment.sh``

    #!/bin/bash

    export TF_CPP_MIN_LOG_LEVEL=3
    export TF_XLA_FLAGS=--tf_xla_enable_xla_devices

    module load miniconda-3

    # Activate installed conda environment with DeepHyper
    conda activate /lus/grand/projects/datascience/regele/theta/test/dhcpu/

Once the ``init-dh-environment.sh`` is created we need to define a submission script. The goal of this script is to (1) request a given amount of ressources, (2) launch a Ray cluster accross all compute nodes, (3) execute a DeepHyper task which distribute the computation on the Ray workers. Create a ``deephyper-job.qsub`` script:

.. code-block::

    mkdir exp && cd exp/
    touch deephyper-job.qsub

Then add the following content:

.. code-block:: bash

    #!/bin/bash
    #COBALT -A datascience
    #COBALT -n 2
    #COBALT -q debug-flat-quad
    #COBALT -t 30
    #COBALT --attrs enable_ssh=1

    # User Configuration
    EXP_DIR=$PWD
    INIT_SCRIPT=$PWD/init-dh-environment.sh

    # Initialize the head node
    source $INIT_SCRIPT

    # Collect IP addresses of compute nodes
    nodes_array=($(python -m deephyper.core.cli.nodelist theta $COBALT_PARTNAME | grep -P '\[.*\]' | tr -d '[],'))
    HEAD_NODE_IP=${nodes_array[0]}
    WORKER_NODES_IPS=${nodes_array[@]:1}

    # Create YAML configuration for the Ray cluster
    # Each compute node will have 2 Ray workers (--num-cpus 2)
    deephyper ray-cluster config --init $INIT_SCRIPT --head-node-ip $HEAD_NODE_IP --worker-nodes-ips ${WORKER_NODES_IPS[@]} --num-cpus 2 -v

    # Launch the Ray cluster
    ray up ray-config.yaml -y

    # Execute the DeepHyper Task
    # Here the task is an hyperparameter search using the DeepHyper CLI
    # However it is also possible to call a Python script using different
    # Features from DeepHyper (see following notes)
    ssh $HEAD_NODE_IP "source $INIT_SCRIPT; cd $EXP_DIR; \
        deephyper hps ambs \
        --problem deephyper.benchmark.nas.linearRegHybrid.Problem \
        --evaluator ray \
        --run-function deephyper.nas.run.quick.run \
        --ray-address auto \
        --ray-num-cpus-per-task 1

    ray down ray-config.yaml -y

.. warning::

    The ``#COBALT --attrs enable_ssh=1`` is crucial otherwise ``ssh`` calls will be blocked by the system.

.. tip::

    The different ``#COBALT`` arguments can also be passed through the command line:

    .. code-block:: console

        qsub -n 2 -q debug-flat-quad -t 30 -A datascience \
            --attrs enable_ssh=1 \
            deephyper-job.qsub


.. admonition:: Use a Python script instead of DeepHyper CLI
    :class: dropdown

    Instead of calling ``deephyper hps ambs`` in ``deephyper-job.qsub`` it is possible to call a custom Python script with the following content:

    .. code-block:: python
        :caption: **file**: ``myscript.py``

        def run(hp):
            return hp["x"]

        if __name__ == "__main__":
            import os
            from deephyper.problem import HpProblem
            from deephyper.search.hps import AMBS
            from deephyper.evaluator.evaluate import Evaluator

            problem = HpProblem()
            problem.add_hyperparameter((0.0, 10.0), "x")

            evaluator = Evaluator.create(
                run, method="ray", method_kwargs={
                    "address": "auto"
                    "num_cpus_per_task": 1
                }
            )

            search = AMBS(problem, evaluator)

            search.search()

    Then replace the ``ssh`` call with:

    .. code-block:: bash

        ssh $HEAD_NODE_IP "source $INIT_SCRIPT; cd $EXP_DIR; \
            python myscript.py"

    This can be more practical to use this approach when integrating DeepHyper in a different workflow.


Strategy 2 - 1 evaluation per multiple nodes
============================================

The Ray workers are launch on the head node this time. This will allow us to use MPI inside our run-function.

.. code-block:: bash
    :caption: **file**: ``deephyper-job.qsub``

    #!/bin/bash
    #COBALT -A datascience
    #COBALT -n 2
    #COBALT -q debug-flat-quad
    #COBALT -t 30

    # Initialize the head node
    EXP_DIR=$PWD
    INIT_SCRIPT=$PWD/SetUpEnv.sh
    source $INIT_SCRIPT

    # Start Ray workers on the head node
    for port in $(seq 6379 9000); do
        RAY_PORT=$port;
        ray start --head --num-cpus 2 --port $RAY_PORT;
        if [ $? -eq 0 ]; then
            break
        fi
    done

    # Execute the DeepHyper Task
    python myscript.py

In this case the ``run`` function can call MPI routines:

.. code-block:: python

    import os

    def run(config):

        os.system("aprun -n .. -N ..")

        return parse_result()


