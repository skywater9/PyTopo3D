#!/bin/bash
# Run topology optimization with the obstacle configuration

# Run with the default example config file
python main.py --nelx 60 --nely 20 --nelz 10 --obstacle-config examples/obstacles_config_cylinder.json \
    --experiment-name "cylinder_obstacle_test" \
    --description "Topology optimization with cylindrical obstacle in the center" \
    --tolx 0.1 --maxloop 1000