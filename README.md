# PyTorch Robot Kinematics
- Parallel forward kinematics (FK)
- (TODO test this) Differentiable FK
- Load robot description from URDF, SDF, MJCF file 

# Usage
Clone repository somewhere, then `pip3 install -e .` to install in editable mode.

```python
import torch
import pytorch_kinematics as pk

# load robot description from URDF formatted string

```

# Credits
- `pytorch_kinematics/transforms` is extracted from [pytorch3d](https://github.com/facebookresearch/pytorch3d) with only minor edits.
This was done instead of including `pytorch3d` as a dependency because it is hard to install and most of its code is unrelated.
- `pytorch_kinematics/urdf_parser_py`, and `pytorch_kinematics/mjcf_parser` is extracted from [kinpy](https://github.com/neka-nat/kinpy), as well as the FK logic.
The contribution of this repository is mainly to port the logic to pytorch and parallelize it.
