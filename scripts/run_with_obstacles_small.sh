#!/bin/bash
# Run topology optimization with the obstacle configuration

# Run with the default example config file but with reduced element count for faster execution
python main.py --nelx 30 --nely 15 --nelz 10 --obstacle-config examples/obstacles_config_cylinder.json -tolx 0.1 --maxloop 1000