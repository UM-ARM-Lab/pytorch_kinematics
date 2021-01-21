# PyTorch Robot Kinematics
- Parallel forward and inverse kinematics (FK and IK)
- (TODO test this) Differentiable FK
- Load robot description from URDF, SDF, and MJCF file 

# Usage
Clone repository somewhere, then `pip3 install -e .` to install in editable mode.

# Transform Library
The `pytorch_kinematics/transforms` is extracted from `pytorch3d` with only minor edits.
This was done instead of including `pytorch3d` as a dependency because it is hard to install and most of its code is unrelated.
