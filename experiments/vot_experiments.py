"""Run VOT experiments."""

import os.path
import got10k.experiments
import tracking.trackerADMDNet

configurations = [
    {"grl": "constant"},
    {"grl_direction": "decreasing", "grl": "arccosine"},
    {"grl_direction": "decreasing", "grl": "cosine_annealing"},
    {"grl_direction": "decreasing", "grl": "exponential"},
    {"grl_direction": "decreasing", "grl": "gamma"},
    {"grl_direction": "decreasing", "grl": "linear"},
    {"grl_direction": "decreasing", "grl": "pada"},
    {"grl_direction": "increasing", "grl": "arccosine"},
    {"grl_direction": "increasing", "grl": "cosine_annealing"},
    {"grl_direction": "increasing", "grl": "exponential"},
    {"grl_direction": "increasing", "grl": "gamma"},
    {"grl_direction": "increasing", "grl": "linear"},
    {"grl_direction": "increasing", "grl": "pada"},
]

experiment = got10k.experiments.ExperimentVOT(
    os.path.expanduser("~/Videos/vot-got"), version=2019, experiments="supervised",
)
experiment.repetitions = 5
tracker_names = []
for configuration in configurations:
    if "grl_direction" in configuration:
        name = configuration["grl_direction"][0:3] + "_" + configuration["grl"]
    else:
        name = configuration["grl"]
    tracker_names.append(name)
    tracker = tracking.trackerADMDNet.ADMDNet(name, configuration)
    experiment.run(tracker)
experiment.report(tracker_names)
