Running TMFT
============

I assume you have Python and Pip installed on your system.
TMFT requires Python 3.

#. Clone this repository, or fork it in Github and clone your fork.
#. Install dependencies.

   .. code-block:: bash

     pip install --requirement requirements.txt

#. Download the [MDNet ImageNet-Vid model](https://github.com/hyeonseobnam/py-MDNet/raw/master/models/mdnet_imagenet_vid.pth).
   Save it to *models/* in your repository.

Running the Smoke Test
======================

The smoke test provides a quick indication if changes to the code cause errors or degrade performance.
To run the smoke test, launch the Smoke Test debug configuration in VS Code, or run this command

.. code-block:: bash

   python3 -m experiments.run --smoke-test Deer

.. warning::

   The smoke test seeds the random number generators with 0.
   This makes the smoke test deterministic so we know that performance changes are due to code changes.

Original MDNet-based Algorithm
------------------------------

========  =============================================================  ========
Sequence  LR Schedule                                                    Mean IoU
========  =============================================================  ========
Deer      Inc. Exponential (g=0.726, lr\ :sub:`max`\ =1.0)               0.714
Deer      Inc. PADA (l=1, a=10, lr\ :sub:`min`\ =0, lr\ :sub:`max`\ =1)  0.726 
Deer      Constant (c=1.0)                                               0.713
========  =============================================================  ========

Cleaned Up TMFT
---------------

========  =============================================================  ========
Sequence  LR Schedule                                                    Mean IoU
========  =============================================================  ========
Deer      Inc. PADA (l=1, a=10, lr\ :sub:`min`\ =0, lr\ :sub:`max`\ =1)  0.726 
========  =============================================================  ========

Running OTB Experiments
=======================

Use :py:mod:`experiments.otb` script to run OTB-50 and OTB-100 experiments.
The module uses the GOT-10k tool to run the experiments and produce the report.

.. code-block:: bash

  usage: otb.py [-h] [--version {tb50,tb100}] [--tracker-name TRACKER_NAME]
                [--dataset-path DATASET_PATH] [--result-path RESULT_PATH]
                [--report-path REPORT_PATH]

  Run OTB experiments.

  optional arguments:
    -h, --help            show this help message and exit
    --version {tb50,tb100}
                          The OTB dataset to use. (default: tb100)
    --tracker-name TRACKER_NAME
                          The tracker name to use in experiment results and
                          reports. (default: TMFT)
    --dataset-path DATASET_PATH
                          The path to the dataset on disk. (default:
                          ~/Videos/otb)
    --result-path RESULT_PATH
                          The path to write tracking results. (default:
                          ~/repositories/tmft/results)
    --report-path REPORT_PATH
                          The path to write experiment reports. (default:
                          ~/repositories/tmft/reports)

Running UAV123 Experiments
==========================

Use :py:mod:`experiments.uav` script to run UAV123 experiments.
The module uses the GOT-10k tool to run the experiments and produce the report.

.. code-block:: bash

  usage: uav.py [-h] [--version {UAV123,UAV20L}] [--tracker-name TRACKER_NAME]
                [--dataset-path DATASET_PATH] [--result-path RESULT_PATH]
                [--report-path REPORT_PATH]

  Run UAV123 experiments.

  optional arguments:
    -h, --help            show this help message and exit
    --version {UAV123,UAV20L}
                          The UAV dataset to use. (default: UAV123)
    --tracker-name TRACKER_NAME
                          The tracker name to use in experiment results and
                          reports. (default: TMFT)
    --dataset-path DATASET_PATH
                          The path to the dataset on disk. (default:
                          ~/Videos/uav123)
    --result-path RESULT_PATH
                          The path to write tracking results. (default:
                          ~/repositories/tmft/results)
    --report-path REPORT_PATH
                          The path to write experiment reports. (default:
                          ~/repositories/tmft/reports)

Running VOT Experiments
=======================

Use :py:mod:`experiments.vot` script to run VOT experiments.
The module uses the GOT-10k tool to run the experiments and produce the report.

.. code-block:: bash

  usage: vot.py [-h] [--version {2013,2014,2015,2016,2017,2018}]
                [--tracker-name TRACKER_NAME] [--dataset-path DATASET_PATH]
                [--result-path RESULT_PATH] [--report-path REPORT_PATH]

  Run VOT experiments.

  optional arguments:
    -h, --help            show this help message and exit
    --version {2013,2014,2015,2016,2017,2018}
                          The VOT dataset to use. (default: 2018)
    --tracker-name TRACKER_NAME
                          The tracker name to use in experiment results and
                          reports. (default: TMFT)
    --dataset-path DATASET_PATH
                          The path to the dataset on disk. (default:
                          ~/Videos/vot-got)
    --result-path RESULT_PATH
                          The path to write tracking results. (default:
                          ~/repositories/tmft/results)
    --report-path REPORT_PATH
                          The path to write experiment reports. (default:
                          ~/repositories/tmft/reports)
