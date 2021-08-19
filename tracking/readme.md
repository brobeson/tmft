# Running Experiments

To run experiments, a configuration file is required. See [Configuring
Experiments](configuring-experiments) for details on the configuration file.

Once you have the configuration file written, run the experiment runner using Python 3:

```bash
python3 -m experiments.run
```

Use `--help` to view complete command line options:

```bash
python3 -m experiments.run --help`
```

---

## Configuring Experiments

Experiments are configured with a YAML file. The default configuration file is
*configuration.yaml*. An alternative file can be specified on the command line:

```bash
python3 -m experiments.run baseline.yaml
```

### Configuring TMFT

One of the root elements of the configuration file must the object `tmft`. This
table describes the TMFT options.

| Option      | Required | Type    | Default | Description |
|:------------|:---------|:--------|:--------|:---|
| name        | no       | string  | TMFT    | The name to use for writing experiment results. |
| random_seed | no       | int     | n/a     | A value to seed the random generators used by TMFT. If this is omitted, the generators are not explicitly seeded. **Note that this option is ignored for VOT experiments.** The VOT protocol runs each sequence multiple times, so the tracker must remain stochastic. |
| use_gpu     | yes      | boolean |         | `true` indicates to use the GPU. Doing so requires a CUDA capable graphics processor. |

#### Domain Adaptation Learning Rate Schedule

TMFT supports multiple learning rate schedules for the domain adaptation
gradient reverse layer (GRL). To specify the GRL, set the `grl` option.
Possible values are

- `constant`
- `cosine_annealing`
- `gamma`
- `inverse_cosine_annealing`
- `linear`
- `pada`

The schedule must be configured by specifying an object with the same name as
the `grl` value. The tables below describe the options for each schedule. All
the options are required.

##### Constant

| Option | Type | Description |
|:-------|:-----|:------------|
| constant | float | The constant value to use for the learning rate schedule. |

##### Cosine Annealing

| Option       | Type    | Description                                                         |
|:-------------|:--------|:--------------------------------------------------------------------|
| epochs       | integer | The maximum number of training epochs for each online update phase. |
| maximum_rate | float   | The maximum allowed learning rate for the schedule.                 |
| minimum_rate | float   | The minimum allowed learning rate for the schedule.                 |

##### Gamma

| Option       | Type    | Description                                                         |
|:-------------|:--------|:--------------------------------------------------------------------|
| epochs       | integer | The maximum number of training epochs for each online update phase. |
| gamma        | float   | The gamma parameter for the schedule.                               |
| maximum_rate | float   | The maximum allowed learning rate for the schedule.                 |
| minimum_rate | float   | The minimum allowed learning rate for the schedule.                 |

##### Inverse Cosine Annealing

The options for inverse cosine annealing are identical to the [cosine
annealing](#cosine-annealing) option.

##### Linear

| Option       | Type    | Description                                                         |
|:-------------|:--------|:--------------------------------------------------------------------|
| epochs       | integer | The maximum number of training epochs for each online update phase. |
| maximum_rate | float   | The maximum allowed learning rate for the schedule.                 |
| minimum_rate | float   | The minimum allowed learning rate for the schedule.                 |

##### PADA

| Option       | Type    | Description                                                         |
|:-------------|:--------|:--------------------------------------------------------------------|
| alpha        | float   | The value of the equation's α parameter.                            |
| epochs       | integer | The maximum number of training epochs for each online update phase. |
| lambda       | float   | The value of the equation's λ parameter.                            |
| maximum_rate | float   | The maximum allowed learning rate for the schedule.                 |
| minimum_rate | float   | The minimum allowed learning rate for the schedule.                 |

#### TODO

> Finish documenting the TMFT options.

### Smoke Test Configuration

The experiment runner can run a smoke test. The smoke test runs a single
sequence and compares the tracking results with known prior results. To run a
smoke test, include the section `smoke_test` in the configuration file.
This table describes the available smoke tests:

| Option      | Required | Type    | Default | Description                                                                                      |
|:------------|:---------|:--------|:--------|:-------------------------------------------------------------------------------------------------|
| frame_count | no       | integer | n/a     | The number of frames to run from the sequence. If this is omitted, the entire sequence will run. |
| save_times  | no       | boolean | n/a     | If present, and set to `true`, the tracking times for each frame will be written to a file.      |
| skip        | no       | boolean | n/a     | If present, and set to `true`, the smoke test will be skipped.                                   |

#### Warning

> The smoke test forces the TMFT random seed to a consistent value between runs. This is necessary for the tracking results comparison to be accurate.

### GOT-10k Configuration

To run experiments with the GOT-10k tool, add the section `got10k_experiments`.
The following options are allowed in this section.

| Option  | Required | Type    | Default | Description                                                    |
|:--------|:---------|:--------|:--------|:---------------------------------------------------------------|
| display | no       | boolean | false   | Whether to display imagery and tracking results on the screen. |

In addition, at least one experimental dataset must be specified.

#### OTB

[OTB](http://www.visual-tracking.net/) experiments are specified in the `otb` section.

| Option     | Required | Type    | Default       | Description                                         |
|:-----------|:---------|:--------|:--------------|:----------------------------------------------------|
| display    | no       | boolean | n/a           | Override the value of `got10k_experiments.display`. |
| result_dir | no       | string  | ./results/otb | Where to write tracking results.                    |
| root_dir   | yes      | string  |               | The root path to the OTB sequences.                 |
| skip       | no       | boolean | `false`       | This can be used to skip the OTB experiments. This is useful if you want to run experiments on a different dataset, but don't want to delete the OTB configuration. |
| version    | yes      | string  |               | The OTB experiments and datasets to run. The available options are `2013`, `2015`, `tb50`, and `tb100`. This option can also be a list of datasets. |
| save_loss  | no       | boolean | `false`       | If `true`, TMFT will be instructed to save loss data during training, and the experiment runner will save it to *{result_dir}/{name}/{sequence}.txt*. The *{name}* portion is the TMFT tracker name. |

#### VOT

[VOT](https://votchallenge.net/) experiments are specified in the `vot` section.

| Option     | Required | Type    | Default       | Description                                                                    |
|:-----------|:---------|:--------|:--------------|:-------------------------------------------------------------------------------|
| display    | no       | boolean | n/a           | Override the value of `got10k_experiments.display`.                            |
| result_dir | no       | path    | ./results/vot | Where to write tracking results.                                               |
| root_dir   | yes      | path    |               | The root path to the VOT sequences.                                            |
| skip       | no       | boolean | `false`       | This can be used to skip the VOT experiments. This is useful if you want to run experiments on a different dataset, but don't want to delete the VOT configuration. |
| version    | yes      |         |               | The VOT experiments and datasets to run. **Only 2016 is supported right now.** |

#### Example GOT-10k Configuration

```yaml
got10k_experiments:
   display: false             # Don't display imagery during experiments.
   otb:
      version: 2015           # Run experiments on the OTB 2015 dataset.
      root_dir: ~/Videos/otb  # OTB sequences are in ~/Videos/otb.
   vot:
      version: 2016           # Only run VOT 2016 experiments.
      root_dir: ~/Videos/vot  # Sequences are in ~/Videos/vot.
      display: true           # Display the VOT imagery.
```

## Running Multiple Experiments Unattended

The experiment runner also supports a multi-configuration YAML file. The
configuration file specifies experimental configurations to run. The format of the multi-configuration file is:

| Option         | Required | Type            | Default | Description                                                   |
|:---------------|:---------|:----------------|:--------|:--------------------------------------------------------------|
| multirun       | yes      | object          |         | Informs the experiment runner to run multiple experiments.    |
| path           | no       | string          | ./      | The directory in which to find the experiment configurations. |
| configurations | yes      | list of strings |         | The list of experiment configurations. If an entry is an absolute path, it is used as-is. Otherwise, the entry is appended to the value of `path`. In all cases, ".yaml" is appended if it is not present. |

Here is an example multi-experiment configuration.

```yaml
multirun:
   path: ~/experiments/configurations
   configurations:
      - configuration_a
      - configuration_b
      - ~/baseline/baseline.yaml
```

In this example, three experiments are run with the following configuration
files:

- ~/experiments/configurations/configuration_a.yaml
- ~/experiments/configurations/configuration_b.yaml
- ~/baseline/baseline.yaml

---

## Using the GOT-10k Tool

The [GOT-10k](https://github.com/got-10k/toolkit/) tool can run experiments for
multiple benchmarks and datasets.

1. Ensure you have the GOT-10k tool
   [installed](https://github.com/got-10k/toolkit/#installation).
1. Modify *tracking/options.yaml*, or write your own YAML configuration file for
   your experiments. See [Configuring Experiments](#configuring-experiments) for
   configuration file details.
1. Run the Python file *run_tracking.py*:
   ```bash
   python3 run_tracking.py
   ```
