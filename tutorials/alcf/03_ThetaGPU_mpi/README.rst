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
    module load openmpi/openmpi-4.0.5

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
    COBALT_JOBSIZE=2
    RANKS_PER_NODE=8

    # Initialization of environment
    source $INIT_SCRIPT

    echo "mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE --hostfile $COBALT_NODEFILE python myscript.py";
    mpirun -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH -n $(( $COBALT_JOBSIZE * $RANKS_PER_NODE )) -N $RANKS_PER_NODE --hostfile $COBALT_NODEFILE python myscript.py


Edit the ``#COBALT ...`` directives:

.. code-block:: bash

    #COBALT -A $PROJECT_NAME
    #COBALT -n 2
    #COBALT -q full-node
    #COBALT -t 20
    #COBALT --attrs filesystems=home,grand,eagle,theta-fs0

Along with the ``COBALT_JOBSIZE`` which should always correspond to the ``-n`` attribute from the ``#COBALT ...`` directives.

And adapt the executed Python application depending on your needs:

.. code-block:: python

    myscript.py

Finally, submit the script from a ThetaGPU login node (e.g., ``thetagpusn1``):

.. code-block:: bash

    qsub deephyper-job.qsub

