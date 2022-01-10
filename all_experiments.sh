#!/bin/bash

if [[ $# < 1 ]]
then
  echo "Enter the tracker name"
  exit 1
fi

# Pilot Study
python3 -m experiments.pilot_study \
  --tracker-name "${1}" \
  Car4 Car24 Deer FleetFace Jump
  
# VOT 2019
python3 -m experiments.got10k_experiments \
  --tracker-name "${1}" \
  --dataset-path ~/Videos/vot/2019 \
  --slack-file ~/.slack_channel.yml \
  2019

# OTB-100
python3 -m experiments.got10k_experiments \
  --tracker-name "${1}" \
  --dataset-path ~/Videos/otb \
  --slack-file ~/.slack_channel.yml \
  tb100

# UAV123
python3 -m experiments.got10k_experiments \
  --tracker-name "${1}" \
  --dataset-path ~/Videos/uav123 \
  --slack-file ~/.slack_channel.yml \
  uav123
