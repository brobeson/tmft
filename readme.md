# Temporally Matching Filter Tracker (TMFT)

TMFT is a research project investigating the application of domain adaptation to online updates in single-object trackers.
TMFT is built on [py-MDNet](http://cvlab.postech.ac.kr/research/mdnet/); see the [py-MDNet license](https://github.com/hyeonseobnam/py-MDNet/blob/master/LICENSE) for details regarding reuse of py-MDNet code.

You can read our paper, [Object Tracking Using Matching Filters](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/cvi2.12040), published in *IET Computer Vision*.

If you're using this code for your research, please cite:

```bibtex
@article{robeson2021tmft,
  author  = {Robeson, Brendan and Javanmardi, Mohammadreza and Qi, Xiaojun},
  title   = {Object Tracking Using Temporally Matching Filters},
  journal = {{IET} Computer Vision},
  volume  = {15},
  number  = {4},
  pages   = {245--257},
  doi     = {10.1049/cvi2.12040},
  url     = {https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/cvi2.12040},
  year    = {2021},
  month   = mar
}
```

## Running TMFT

I assume you have Python and Pip installed on your system.
TMFT requires Python 3.

1. Clone this repository, or fork it in Github and clone your fork.
1. Install dependencies.
   ```bash
   pip install --requirement requirements.txt
   ```
1. Download the [MDNet ImageNet-Vid model](https://github.com/hyeonseobnam/py-MDNet/raw/master/models/mdnet_imagenet_vid.pth).
   Save it to *models/* in your repository.

### Running a Smoke Test

The smoke test provides a quick indication if changes to the code cause errors or degrade performance.
Before running the smoke test, ensure the `random_seed` is set to 0 in [tracking/options.yaml](tracking/options.yaml).
To run the smoke test, run `python3 -m tracking.run_tracker --seq Deer`, or launch the Smoke Test debug configuration in VS Code.

#### Original MDNet-based Algorithm

| Sequence | LR Schedule | Mean IoU |
|:---|:---|---:|
| Deer | Inc. Exponential (g=0.726, lr<sub>max</sub>=1.0) | 0.714 |
| Deer | Inc. PADA (l=1, a=10, lr<sub>min</sub>=0, lr<sub>max</sub>=1) | 0.726 |
| Deer | Constant (c=1.0) | 0.713 |

#### Cleaned Up Code

| Sequence | LR Schedule | Mean IoU |
|:---|:---|---:|
| Deer | Inc. PADA (l=1, a=10, lr<sub>min</sub>=0, lr<sub>max</sub>=1) | 0.726 |

---

## Usage

### Tracking

```bash
python tracking/run_tracker.py -s DragonBaby [-d (display fig)] [-f (save fig)]
```

- You can provide a sequence configuration in two ways (see tracking/gen_config.py):
  - `python tracking/run_tracker.py -s [seq name]`
  - `python tracking/run_tracker.py -j [json path]`
