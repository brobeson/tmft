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

## Getting Started

1. Clone the repository.
1. Install [Sphinx](https://www.sphinx-doc.org).
1. In the [docs/](docs/) directory, run `make html`.
1. Open *docs/build/html/index.html* in your web browser.
