Configuring TMFT
================

Configure TMFT with the YAML file *tracking/options.yaml*.
This table describes the configuration options.

.. list-table:: TMFT Configuration Options
   :widths: auto
   :header-rows: 1

   * - Option
     - Type
     - Description
   * - ``use_gpu``
     - boolean
     - [**Deprecated**] TMFT automatically uses the GPU if one is available.
     
       Indicates whether to use the GPU. A CUDA capable GPU is required for this option.
   * - ``model_path``
     - string
     - The path to the model weights. TMFT loads this data during initialization.
