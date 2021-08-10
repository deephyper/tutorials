.. _tutorial-08:

Neural Architecture Search with Multiple Input Tensors
******************************************************

.. warning::

    Be sure to work in a virtual environment where you can easily ``pip install`` new packages. This typically entails using either Anaconda, virtualenv, or Pipenv.

    Some parts of this tutorial requires pydot (``pip install pydot``) and graphviz (see installation instructions at `https://graphviz.gitlab.io/download/ <https://graphviz.gitlab.io/download/>`_).

In this tutorial we will extend on the previous basic NAS tutorial (:ref:`tutorial-04`) to allow for varying numbers of input tensors. This calls for the construction of a novel search space where the different input tensors may be connected to any of the variable node operations within the search space. The data use for this tutorial is provided in this repository and is a multidelity surrogate modeling data set obtained from the `Brannin function <https://www.sfu.ca/~ssurjano/branin.html>`_. In addition, to the independent variables for this modeling task, low and medium fidelity estimates of the output variable are used as additional inputs to the eventual high fidelity surrogate. Thus, this requires multiple input tensors which may independently or jointly interact with the neural architecture.

Setting up the problem
======================

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

We can now define DeepHyper NAS search problems inside this directory, using
``deephyper new-problem nas {name}``.

Let's set up a NAS problem called ``multi_input_search`` as follows:

.. code-block:: console
    :caption: bash

    cd dh_project/dh_project/
    deephyper new-problem nas multi_input_search


A new NAS problem subdirectory should be in place. This is a Python subpackage containing
sample code in the files ``__init__.py``, ``load_data.py``, ``search_space.py``, and ``problem.py``. Overall, your project directory should look like:

.. code-block::

    dh_project/
        dh_project/
            __init__.py
            multi_input_search/
                __init__.py
                load_data.py
                search_space.py
                problem.py
        setup.py

Load the data
=============

First, we will look at the ``load_data.py`` file that loads and returns the training and validation data from the multifidelity `Brannin function <https://www.sfu.ca/~ssurjano/branin.html>`_. This data set is provided in the `deephyper/tutorials repository <https://github.com/deephyper/tutorials>`_.

.. literalinclude:: dh_project/dh_project/multi_input_search/load_data.py
    :caption: multi_input_search/load_data.py
    :linenos:

The output interface of the ``load_data`` function is important when you have several inputs or outputs. In this case, for the inputs we have a list of 3 numpy arrays.

Define a neural architecture search space
=========================================

Then, we will take a look at ``search_space.py`` which contains the code for
the neural network search_space definition.

.. literalinclude:: dh_project/dh_project/multi_input_search/search_space.py
    :linenos:
    :caption: multi_input_search/search_space.py

Visualize a randomly generated neural network from this search space:

.. code-block::

    python search_space.py

.. image:: _static/random_model.png
    :scale: 50 %
    :alt: random model from regression search space
    :align: center

You can notice the 3 inputs tensors (``input_0, input_1, input_2``).


Create a problem instance
=========================

Now, we will take a look at ``problem.py`` which contains the code for the
problem definition.

.. literalinclude:: dh_project/dh_project/multi_input_search/problem.py
    :linenos:
    :caption: multi_input_search/problem.py

You can look at the representation of your problem:

.. code-block:: console
    :caption: bash

    python problem.py

The expected output is:

.. code-block:: console

    Problem is:
    * SEED = 2019 *
    - search space   : dh_project.multi_input_search.search_space.create_search_space
    - data loading   : dh_project.multi_input_search.load_data.load_data
    - preprocessing  : None
    - hyperparameters:
        * verbose: 0
        * batch_size: 64
        * learning_rate: 0.001
        * optimizer: adam
        * num_epochs: 200
        * callbacks: {'EarlyStopping': {'monitor': 'val_r2', 'mode': 'max', 'verbose': 0, 'patience': 5}, 'ModelCheckpoint': {'monitor': 'val_loss', 'mode': 'min', 'save_best_only': True, 'verbose': 0, 'filepath': 'model.h5', 'save_weights_only': False}}
    - loss           : mse
    - metrics        :
        * r2
    - objective      : val_r2
    - post-training  : None


Execute the search locally
==========================

Everything is ready to run. Let's remember the structure of our experiment:

.. code-block::

    multi_input_search/
        __init__.py
        load_data.py
        problem.py
        search_space.py

Each of these files have been tested one by one on the local machine. Next, we will run a random search (RDM).

.. code-block:: console
    :caption: bash

    deephyper nas random --evaluator ray --problem dh_project.multi_input_search.problem.Problem --max-evals 10 --num-workers 2

.. note::

    In order to run DeepHyper locally and on other systems we are using :mod:`deephyper.evaluator`. For local evaluations we can use the :class:`deephyper.evaluator.RayEvaluator` or the :class:`deephyper.evaluator.SubProcessEvaluator`.


After the search is over, you will find the following files in your current folder:

.. code-block:: console

    deephyper.log
    init_infos.json
    results.csv
    save/

One may visualize the training of the models and the best architectures obtained by following the same procedure as Tutorial 4.