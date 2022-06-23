Introduction to DBO
===================

**Author(s)**: Joceran Gouneau & Romain Egele.

In this tutorial we show how to use the Distributed Bayesian Optimization (DBO) search algorithm to perform hyperparameter optimization on the Ackley function.

Definition of the problem : the Ackley function
-----------------------------------------------

.. image:: https://www.sfu.ca/~ssurjano/ackley.png
   :width: 400
   :alt: Ackley Function in 2D
   :align: center

.. math::

   f(x) = -a \exp \left( -b \sqrt {\frac 1 d \sum_{i=1}^d x_i^2} \right) - \exp \left( -b \sqrt {\frac 1 d \sum_{i=1}^d \cos(c x_i)} \right) + a + \exp(1)

.. note::

   We are using this function to emulate a realistic problem while keeping the definition of the hyperparameter search space and run function as simple as possible ; if you are searching for neural network use cases we redirect you to our Colab and Notebook tutorials.

First we have to define the Hyperparameter search space as well as the run function, which, given a certain ``config`` of hyperparameters, should return the objective we want to maximize. We are computing the 10-D (:math:`d = 10`) Ackley function with :math:`a = 20`, :math:`b = 0.2` and :math:`c = 2\pi` and want to find its minimum :math:`f(x=(0, \dots , 0)) = 0` on the domain :math:`[-32.768, 32.768]^{10}`.
Thus we define the hyperparameter problem as :math:`x_i \in [-32.768, 32.768]~ \forall i \in [|0,9|]` and the objective returned by the ``run`` function as :math:`-f(x)`.

.. literalinclude:: ackley.py
   :language: python
   :caption: **file**: ``ackley.py``

Definition of the distributed Bayesian optimization search (DBO)
----------------------------------------------------------------

.. image:: figures/cbo_vs_dbo.jpg
   :scale: 30%
   :alt: Centralized (left) and Distributed (right) architectures
   :align: center

DBO (*right*) is very similar to Centralized Bayesian Optimization (CBO) (*left*) in the sense that we iteratively generate new configurations with an optimizer :math:`O`, evaluate them on Workers :math:`W` by calling the black-box function :math:`f` which takes :math:`t_{eff}` time to be computed, and fit the optimizer on the history of the search (the configuration/objective pairs) to generate better configurations.
The only difference is that with CBO the fitting of the optimizer and generation of new configurations is centralized on a Manager :math:`M`, while with DBO each worker has its own optimizer and these operations are parallelized.
This difference makes DBO a preferable choice when the run function is relatively fast and the number of workers increasing; with a large enough number of workers the fit of the optimizer (which has to be performed each time we generate a configuration) starts to take more time than the run function takes to be evaluated : at that point we obtain a congestion on the manager and  therefore workers become idle because waiting for a new configuration to be evaluated. DBO avoid this type of congestion issue by attributing one optimizer per worker and performing asynchronous communication between optimizers.

DBO can be formally described in the following algorithm:

.. image:: figures/dbo_algo.jpg
   :scale: 35%
   :alt: DBO algorithm
   :align: center

Running DBO
___________

Unlike CBO, DBO doesn't use any evaluator instance, everything is comprised within the DBO search instance and is only compatible with MPI for now:

.. code-block:: python
   :caption: **file**: ``search.py``

   from ackley import hp_problem, run

   search = DBO(
      hp_problem,
      run,
      sync_communication=False,
      log_dir=".",
      checkpoint_file="results.csv",
      checkpoint_freq=1,
   )

.. note::

   The ``checkpoint_file`` and ``checkpoint_freq`` parameters can be used to regularly save the state of the search while it is being performed, this is equivalent to saving the ``results`` dataframe returned by the search in the file named ``checkpoint_file`` within the directory ``log_dir`` at a frequency ``checkpoint_freq``. It is good practice to perform this checkpoint in the case of an issue during the search in order to still have results even though the search doesn't successfuly terminate. By default the results are saved at each iteration in the ``results.csv`` of the current directory.

.. note::

   The ``sync_communication`` parameter, when set to ``True``, allows to force the workers to broadcast their new evaluations synchronously, in this case the frequency at which this broadcast is performed can be specified with the ``sync_communication_freq`` parameter. For example with ``sync_communication=True`` and ``sync_communication_freq=10``, each worker will perform 10 evaluations and wait for every other worker to do as much before broadcasting these new evaluations, and then repeat this process until the end of the search.

Execution of the search : using MPI
-----------------------------------

The backend of DBO uses MPI, so if we want for example to get the final ``results`` and save it at a specific location, we need to get it from a single worker while executing the search for the same amount of time on each one of them :

.. code-block:: python
   :caption: **file**: ``search.py``

   from mpi4py import MPI

   comm = MPI.COMM_WORLD
   rank = comm.Get_rank()

   timeout = 10
   if rank == 0:
      results = search.search(timeout=timeout)
      results.to_csv("results.csv")
   else:
      search.search(timeout=timeout)


Finaly, we can run the ``search.py`` script with MPI:

.. code-block:: console
   
   $ mpirun -np 10 python search.py
