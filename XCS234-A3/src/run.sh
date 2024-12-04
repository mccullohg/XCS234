#!/bin/bash

# Cartpole without baseline
python run.py --config_filename cartpole_no_baseline

# Cartpole with baseline
python run.py --config_filename cartpole_baseline

# Pendulum without baseline
python run.py --config_filename pendulum_no_baseline

# Pendulum with baseline
python run.py --config_filename pendulum_baseline

# Cheetah without baseline: runs around 5h on the provided VM
python run.py --config_filename cheetah_no_baseline

# Cheetah with baseline: runs around 5h on the provided VM
python run.py --config_filename cheetah_baseline
