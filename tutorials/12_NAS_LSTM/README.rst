.. _tutorial-12:

Neural Architecture Search for LSTM Neural Networks
***************************************************

.. warning::

    Be sure to work in a virtual environment where you can easily ``pip install`` new packages. This typically entails using either Anaconda, virtualenv, or Pipenv.

In this tutorial example, we wil recreate results from our recent paper on LSTM search for surrogate modeling of geophysical flows (DOI:10.1109/SC41405.2020.00012).

Setting up the problem
=======================

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

We can now define our neural architecture search problem inside this directory. Let's set up a NAS problem called ``lstm_search`` as follows:

.. code-block:: console
    :caption: bash

    cd dh_project/dh_project/
    deephyper new-problem nas lstm_search


A new NAS problem subdirectory should be in place. This is a Python subpackage containing
sample code in the files ``__init__.py``, ``load_data.py``, ``search_space.py``, and ``problem.py``. Overall, your project directory should look like:

.. code-block::

    dh_project/
        dh_project/
            __init__.py
            lstm_search/
                __init__.py
                load_data.py
                search_space.py
                problem.py
        setup.py

.. Function modifications and training data
.. =========================================

.. The training data for this tutorial is provided in the tutorials repository on github in ``12_NAS_LSTM/dh_project/dh_project/lstm_search/`` and the ``load_data.py``, ``search_space.py``, ``problem.py`` files in the same location can be used to update the templates created in the previous step.

Load the data
=============

.. todo:: explain how the data were created.

Download the data:

.. literalinclude:: dh_project/dh_project/lstm_search/download_data.sh
    :caption: lstm_search/download_data.sh

.. todo::

    * why are the test data used as validation data?
    * why is the preprocessing fit on both? Should be fit on trained data only?

Transform and preprocess the data:

.. literalinclude:: dh_project/dh_project/lstm_search/load_data.py
    :caption: lstm_search/load_data.py

Test the function with ``python load_data.py`` and get the following output:

.. code-block::

    Train Shapes:  (900, 8, 5) (900, 8, 5)
    Valid Shapes:  (211, 8, 5) (211, 8, 5)


Performance of a baseline LSTM
==============================

.. todo:: show the performance of a baseline lstm

.. literalinclude:: dh_project/dh_project/lstm_search/baseline_lstm.py
    :caption: lstm_search/baseline_lstm.py


Define a neural architecture search space
=========================================

.. literalinclude:: dh_project/dh_project/lstm_search/search_space.py
    :caption: lstm_search/search_space.py

An example of a randomly generated architecture from this search space:

.. image:: _static/random_model.png
    :scale: 50 %
    :alt: random model from stacked lstm search space
    :align: center

Create a problem instance
=========================

.. literalinclude:: dh_project/dh_project/lstm_search/problem.py
    :caption: lstm_search/problem.py

Execute the search locally
==========================

Everything is ready to run. Let's remember the structure of our experiment:

.. code-block::

    lstm_search/
        __init__.py
        load_data.py
        problem.py
        search_space.py

Each of these files can also be tested one by one on the local machine (see :ref:`tutorial-04` for details). Next, we will run a random search (RDM).

.. code-block:: console
    :caption: bash

    deephyper nas random --evaluator ray --problem dh_project.lstm_search.problem.Problem --max-evals 10 --num-workers 2

.. note::

    In order to run DeepHyper locally and on other systems we are using :mod:`deephyper.evaluator`. For local evaluations we can use the :class:`deephyper.evaluator.RayEvaluator` or the :class:`deephyper.evaluator.SubProcessEvaluator`.


After the search is over, you will find the following files in your current folder:

.. code-block:: console

    deephyper.log
    init_infos.json
    results.csv
    save/

.. todo::

    * analyse the resutls of the search (plot + topk)
    * shows the best model found

.. code-block:: console
    :caption: bash

    deephyper-analytics topk results.csv -k 3 -o topk.json

To visualize or use this best architecture you can recreated the Keras model this way:

.. literalinclude:: dh_project/dh_project/lstm_search/best_model.py
    :caption: lstm_search/best_model.py

Then, execute the scrip ``python best_model.py`` and visualize the image produced:

.. image:: _static/best_model.png
    :scale: 50 %
    :alt: best model found from stacked lstm search space
    :align: center