Getting TMFT
============

I assume you have Python and Pip installed on your system. TMFT requires
Python 3.

#. Clone this repository, or fork it in Github and clone your fork.
#. Install dependencies.

   .. code-block:: bash

     pip install --requirement requirements.txt

#. Download the `MDNet ImageNet-Vid model
   <https://github.com/hyeonseobnam/py-MDNet/raw/master/models/mdnet_imagenet_vid.pth>`_.
   Save it in the *models/* directory in your repository.

Running a Pilot Study
=====================

A pilot tracking study provides faster feedback at the expense of thoroughness.
The pilot study runs a subset of the `OTB-100 <http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html>`_
dataset. It saves the overlap success data in *pilot_results.json* in the
results directory. Run ``tmft.py pilot`` to run pilot study.

.. literalinclude:: generated/tmft_pilot_help.rst
  :language: text

Pilot Study Database
--------------------

The pilot study saves tracking results in *pilot_results.json* in the results
directory. The root element of the database is a JSON object. Each element in
the root object is a dictionary. The keys are tracker names and the values are
JSON objects with the trackers' results. Each tracker saves overlap success data
for each sequence on the command line. Each score is the arithmetic mean of the
overlap success for each frame in the sequence.

.. code-block:: json

  {
    "TMFT": {
      "Car4": 0.783,
      "Deer": 0.714
    },
    "MDNet": {
      "Car4": 0.745,
      "Deer": 0.710
    }
  }

You can use the ``tmft.py report`` command to create a table of results from the
pilot study database.

Running Tracking Experiments
============================

Run a tracking with the ``tmft.py experiment`` command. This command runs the
tracker on one common tracking benchmark.

.. literalinclude:: generated/tmft_experiment_help.rst
  :language: text

The command saves the tracking results to a subdirectory of the ``results_dir``:
*results_dir/benchmark/tracker_name*. The form of the tracking results depends
on the tracking benchmark.

The ``slack_file`` parameters tells the command to notify a `Slack
<https://www.slack.com>`_ channel when the experiment starts, and when it stops.

Creating Reports
================

Run the ``tmft.py report`` command to create experiment reports.

.. literalinclude:: generated/tmft_report_help.rst
  :language: text

The ``results_dir`` must contain benchmark subdirectories. An example result
directory layout is:

.. code-block:: text

  ~/experiments/results/
    OTBtb100/
    OTBtb50/
    UAV123/
    VOT2018/

In this case, the report command is ``tmft.py report
~/experiments/results/tmft``.

``tmft.py report`` performs these tasks:

#. It finds all the known benchmarks in the ``results_dir``.
#. It finds all the trackers within each benchmark subdirectory.
#. It uses `GOT-10k <https://github.com/got-10k/toolkit>`_ to create
   benchmark-specific reports.
#. It reads the benchmark-specific reports to create a LaTeX table summarizing
   the results.
#. It reads pilot study data from ``results_dir/pilot_results.json`` to
   create a LaTeX table summarizing the pilot study.

The script writes the reports to the ``REPORT_DIR`` directory. Each benchmark
report is in a benchmark/tracker subdirectory:

.. code-block:: text

  ~/experiments/reports/
    OTBtb100/TMFT/
    OTBtb50/TMFT/
    UAV123/TMFT/
    VOT2018/TMFT/
    experiment_summary.tex
    pilot_study.tex
    vot_robustness.tex

The tracker is undefined; it is whichever tracker the script finds first for a
given benchmark. The actual benchmark reports depend on the benchmark.

The *.tex* files are the summary tables produced by the command. Use the
``\include{}`` or ``\input{}`` LaTeX commands to include the tables in a digital
lab notebook or a paper manuscript.
