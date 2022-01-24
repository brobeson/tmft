#!/bin/bash

if [[ $# < 1 ]]
then
  echo "Enter the tracker name"
  exit 1
fi

# Pilot Study
./tmft.py pilot \
  --tracker-name "${1}" \
  Car4 Car24 Deer FleetFace Jump
  
# VOT 2019
./tmft.py experiment \
  --tracker-name "${1}" \
  --dataset-dir ~/Videos/vot/2019 \
  --slack-file ~/.slack_channel.yml \
  2019

# OTB-100
./tmft.py experiment \
  --tracker-name "${1}" \
  --dataset-dir ~/Videos/otb \
  --slack-file ~/.slack_channel.yml \
  tb100

# UAV123
./tmft.py experiment \
  --tracker-name "${1}" \
  --dataset-dir ~/Videos/uav123 \
  --slack-file ~/.slack_channel.yml \
  uav123

# Reports
./tmft.py report
