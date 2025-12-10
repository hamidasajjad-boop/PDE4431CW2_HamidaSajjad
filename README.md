# Articulated 4-DoF Manipulator Simulation (RRRP) — PDE4431 CW2

## Overview
This project implements a simulated articulated 4-DoF manipulator with joint configuration R–R–R–P (three revolute joints and one prismatic joint).
The robot performs a pick-and-place task, lifting objects from the floor and placing them on three shelves at different heights.
The simulation is created using Python and matplotlib and visualizes the manipulator as a simple stick-figure model.

Features:
- DH-based robot parameters (implicit in robot model)
- Forward kinematics (FK)
- Inverse kinematics (IK) that computes joint values for given (x,y,z)
- 3D visualization with buttons for four target positions: Floor, Shelf 1, Shelf 2, Shelf 3
- Smooth interpolation/animation between poses
- Actual end-effector placement used to position objects accurately

## Files
- `robot.py` — robot class, FK & IK
- `simulation.py` — visualization and buttons
- `main.py` — launch script
- `video_script.txt` — suggested voiceover / demo script

## Requirements
- Python 3.8+
- numpy
- matplotlib

## Install dependencies:

python -m pip install numpy matplotlib

## Running the simulation

Launch the program using:

python main.py

A 3D window will appear showing the manipulator, floor objects, shelves, and user interface buttons.

Robot Model (RRRP)

The manipulator uses four joints in the following order:

- Revolute (base rotation)
- Revolute (shoulder rotation in the horizontal plane)
- Revolute (elbow rotation)
- Prismatic (vertical lift for shelf height)

The IK computes planar solutions for x–y using the first three revolute joints and sets the prismatic joint extension to reach the desired z-level.
Unreachable positions are automatically clamped to the nearest reachable point.

## Controls

The four buttons on the left of the simulation window perform:

- Floor 1 — Pick and place object 1 on Shelf 1
- Floor 2 — Pick and place object 2 on Shelf 2
- Floor 3 — Pick and place object 3 on Shelf 3
- Reset — Return the manipulator to its home configuration

## Each pick-and-place operation includes:

- Moving above the object
- Lowering to pick
- Lifting to a safe intermediate height
- Moving above the shelf
- Lowering to place
- Detaching the object
- Returning upward

## Notes

This project demonstrates RRRP manipulator kinematics, forward and inverse kinematics implementation, trajectory interpolation, and a complete animated pick-and-place sequence.

## GitHub Repository

https://github.com/hamidasajjad-boop/PDE4431CW2_HamidaSajjad.git

## Youtube Link
https://youtu.be/ccB4hTLdhYw


## The simulation is intended for PDE4431 Coursework 2.

