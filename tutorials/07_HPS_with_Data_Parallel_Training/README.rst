.. _tutorial-07:

Hyperparameter Search with Data-Parallel Training
*************************************************

Horovod + Balsam (ALCF's Theta)
===============================

.. warning::

    This tutorial shows how to run hyperparameter search with data-parallel training using Horovod and Balsam on the Theta supercomputer at the ALCF. This tutorial follows one of the example provided in the Horovod documentation.

Let's start by creating a new DeepHyper project workspace. This is a directory where you will create search problem instances that are automatically installed and importable across your Python environment.

.. code-block:: console
    :caption: bash

    deephyper start-project dh_project


A new ``dh_project`` directory is created, containing the following files:

.. code-block::

    dh_project/
        dh_project/
            __init__.py
        setup.py

We can now define DeepHyper search problems inside this directory, using ``deephyper new-problem hps {name}`` for HPS. Let's set up an HPS problem called ``mnist`` as follows:

.. code-block:: console
    :caption: bash

    cd dh_project/dh_project/
    deephyper new-problem hps mnist

Start by editing the ``load_data.py``  file. We will download the MNIST dataset and create two functions. One will return the training and test datasets which are useful to estsimate the generalization performance of your model. The other will return a random-split of the full training data that we will call training and validation datasets. The validation dataset is used for the HPS optimization. The script looks like the following:

.. literalinclude:: dh_project/dh_project/mnist/load_data.py
    :linenos:
    :caption: mnist/load_data.py

Then, check that the data are loaded properly by executing the script:

.. code-block:: console
    :caption: bash

    python load_data.py

which should return something like:

.. code-block:: console
    :caption: bash

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 12s 1us/step
    train_X shape: (40200, 28, 28)
    train_y shape: (40200,)
    valid_X shape: (19800, 28, 28)
    valid_y shape: (19800,)


After the data comes the execution of the model for which we want to optimize the hyperparameters. Let's develop a function named ``run`` in the ``model_run.py`` script:

.. literalinclude:: dh_project/dh_project/mnist/model_run.py
    :linenos:
    :caption: mnist/model_run.py

Then, use a script named ``test_horovod.py`` to test the behaviour of Horovod:

.. literalinclude:: dh_project/dh_project/mnist/test_horovod.py
    :linenos:
    :caption: mnist/test_horovod.py

Execute the test script with a MPI command such as  ``horovodrun``, ``mpirun`` or ``aprun``. If you choose ``aprun`` remember to run it from a Theta Mom node:

.. code-block:: console
    :caption: bash

    aprun -n 2 -N 2 python test_horovod.py

.. note::

    Here we assume that you requested a job allocation with `qsub` in interactive mode for example with `qsub -n 2 -q debug-cache-quad -A $PROJECTNAME -t 30 -I`.

    If you want to test with MPI:

    .. code-block::

        mpirun -np 2 python test_horovod.py


Now that we have a function loading the data and a model learning from these data we can create the hyperparameter search problem to define the hyperparameters we want to optimize. Create a ``problem.py`` script:

.. literalinclude:: dh_project/dh_project/mnist/problem.py
    :linenos:
    :caption: mnist/problem.py

Test the problem script in a standalone fashion to make sure there is no error:

.. code-block::

    python problem.py

The next step, is to link our current work to a Balsam database and submit a job to be executed on Theta. Create a Balsam database named ``expdb``:

.. code-block:: console

    balsam init expdb

Then start and connect to the ``expdb`` database:

.. code-block:: console

    source balsamactivate expdb

Finally, we can submit a search job to Balsam and the COBALT scheduler:

.. code-block::

    deephyper balsam-submit hps ambs -w mnist_hvd --problem dh_project.mnist.problem.Problem --run dh_project.mnist.model_run.run -t 30 -q debug-cache-quad -n 4 -A datascience -j mpi --num-nodes-per-eval 2 --num-ranks-per-node 2 --num-threads-per-rank 32