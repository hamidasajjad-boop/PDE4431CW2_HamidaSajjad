# Articulated 4-DoF Manipulator Simulation (RRPR) — PDE4431 CW2

## Overview
This project implements a simulated articulated 4-DoF manipulator (R-R-P-R) to pick objects from the floor and place them on three shelves at different heights. It is a "stick-figure" simulation using Python and `matplotlib`.

Features:
- DH-based robot parameters (implicit in robot model)
- Forward kinematics (FK)
- Inverse kinematics (IK) that computes joint values for given (x,y,z)
- 3D visualization with buttons for four target positions: Floor, Shelf 1, Shelf 2, Shelf 3
- Smooth interpolation/animation between poses

## Files
- `robot.py` — robot class, FK & IK
- `simulation.py` — visualization and buttons
- `main.py` — launch script
- `video_script.txt` — suggested voiceover / demo script

## Requirements
- Python 3.8+
- numpy
- matplotlib

Install dependencies:
```bash
python -m pip install numpy matplotlib
