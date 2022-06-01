Tuning of MPI Programs
======================

**Author(s)**: Denis Boyda.

This tutorial demonstrates the DeepHyper ability to optimize hyperparameters of MPI programs. As a demonstration example, we write simple MPI c++ code and compile it to obtain a binary. When executed the binary initializes MPI, prints some information, and computes a polynomial with parameter obtained through the command line. In DeepHyper we will optimize this binary as black-box function which obtains some parameters through the command line (or input file) and produces a result of executioni. 

This demonstration emulates a situation when a user has a binary that does some computations, and hyperparameters of these computations have to be optimized. In general, one can split binary execution into three stages. In the first, initialization stage, all necessary input files are prepared and a logging directory is created. In the second stage, an MPI program is submitted for execution. And, in the third, finalization stage, output files or artifacts are saved and analyzed, target value obtained from them is returned to DeepHyper.


Example MPI binary compiled from of c++ code
--------------------------------------------

In the c++ source code posted below, we initialize MPI and print information from every worker.  The master worker with rank 0,  additionally evaluates the function ``f`` using the input parameter obtained through command-line argument. Function ``f`` computes polynomial ``f(x)`` as function ``x``. It is a target function in this demonstration, and DeepHyper will optimize it. Solution ``x = - 2`` of this optimization is known analytically for hyperparameters optimization.

.. literalinclude:: mpi_f.c
   :language: c

This code can be compiled as ``mpicc mpi_f.c -o f_exe``. Execution of ``f_exe`` with two MPI workers gives

.. code-block:: console

    $ mpirun -n 2 ./f_exe 2 
    Hello world from processor thetagpu05, rank 0 out of 2 processors
    f(x) = -14.000000
    Hello world from processor thetagpu05, rank 1 out of 2 processors


Python wrapper for MPI binary
-----------------------------

As DeepHyper accepts a Python function as a target function for optimization one needs to prepare a python wrapper for MPI binary. Python used as a scripting language opens large possibilities for such wrappers. In the example below ``run_mpi()``  function is such a wrapper.

.. literalinclude:: search0.py
   :language: python


In ``run_mpi()`` we obtain the absolute path of the MPI binary,  execute it with ``subprocess`` module, and parse the captured output. When submitting to execution in ``supbrosess.run`` we use ``mpirun`` as an executor, specify necessary environment variables and hosts with slots, and add binary ``exe`` with an argument obtained through ``config`` dictionary.  The result of ``exe`` binary execution obtained through parsing is returned from ``run_mpi()``.


More powerfull wrapper with initialization and finalization
-----------------------------------------------------------

In spite of being able to run simple MPI binary this wrapper has several limitations. The simple MPI binary we compiled above obtains hyperparameters through a command-line interface but in general, binary may require input files. Therefore it is important to do some initialization before submitting binary for execution. Another drawback is the absence of finalization. In this demonstration, we create a context manager ``Experiment`` for these purposes. In the initialization phase, ``Experiment`` creates a directory for the run, and changed the correct path to it, under finalization if changes path back. In the created folder we make to two files for ``stdout`` and ``stderr`` produced by binary ``exe``. Running command is also saved. 

Another important change we made is an additional argument ``dequed`` of the `run_mpi` function. This argument should contain a list of available hosts with slots for MPI execution. It allows using different evaluators with an arbitrary number of workers that manage an available resources.

.. literalinclude:: search1.py
   :language: python

When executed this script creates a dirrectory for evaluation, calls binary and saves output with running command.

.. code-block:: console

    $ python search1.py 
    result:  -14.0
    $ ls
    exp-0  f_exe  mpi_f.c  mpi_search.py  __pycache__  search0.py  search1.py  search2.py  search3.py  search.py  tutorial_mpi.ipynb
    $ ls exp-0/
    runcommand.txt	stderr.txt  stdout.txt

Optimization with a single node
-------------------------------

Once the infrastructure for MPI binary was created one can create a DeepHyper problem and run optimization.

.. literalinclude:: search2.py
   :language: python

In this example evaluator does not provide ``dequed`` argument to run the function, therefore, the default value is used. By default we specified only one available host with one slot ``localhost:1``, therefore, one evaluation will be done with only one rank.

Optimization with multiple nodes
--------------------------------

Finally, we demonstrate the execution of a binary within several ranks. For ThetaGPU we obtain a list of nodes in ``get_thetagpu_nodelist()`` and specify that every node has 8 slots. The evaluator is decorated with decorator ``queue`` which manages queue of resources (ThetaGPU nodes in this example) and provides ``queue_pop_per_task`` nodes for one evaluation. In this example ``queue_pop_per_task=2`` such that every evaluation obtain two nodes with eight slots resulting in 16 ranks. The number of workers may be computed by dividing the total number of nodes by the number of nodes per evaluation.  

.. literalinclude:: search3.py
   :language: python