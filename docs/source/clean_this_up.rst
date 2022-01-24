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
