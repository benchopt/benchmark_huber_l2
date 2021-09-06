My BenchOpt Benchmark
=====================
|Build Status| |Python 3.6+|

BenchOpt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to solver of **describe your problem**:

.. math::

  \min_{w, \sigma} {\sum_{i=1}^n\left(\sigma + H_{\epsilon}\left(\frac{X_{i}w - y_{i}}{\sigma}\right)\sigma\right) + \alpha {||w||_2}^2}

where n (or n_samples) stands for the number of samples, p (or n_features) stands for the number of features

.. math::

 X = [x_1^\top, \dots, x_n^\top]^\top \in \mathbb{R}^{n \times p}

and

.. math::

  H_{\epsilon}(z) = \begin{cases}
         z^2, & \text {if } |z| < \epsilon, \\
         2\epsilon|z| - \epsilon^2, & \text{otherwise}
  \end{cases}


Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/benchopt/benchmark_huber_l2
   $ benchopt run benchmark_huber_l2

Apart from the problem, options can be passed to `benchopt run`, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_huber_l2 -s sklearn -d simulated --max-runs 10 --n-repetitions 10

Use `benchopt run -h` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/benchopt/benchmark_huber_l2/workflows/Tests/badge.svg
   :target: https://github.com/benchopt/benchmark_huber_l2/actions
.. |Python 3.6+| image:: https://img.shields.io/badge/python-3.6%2B-blue
   :target: https://www.python.org/downloads/release/python-360/
