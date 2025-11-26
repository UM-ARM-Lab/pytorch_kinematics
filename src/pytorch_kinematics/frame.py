from typing import Any, ClassVar, Iterable, List, Optional

import torch

import pytorch_kinematics.transforms as tf
from pytorch_kinematics.transforms import axis_and_angle_to_matrix_33


class Visual:
    TYPES: ClassVar = ["box", "cylinder", "sphere", "capsule", "mesh"]

    def __init__(
        self,
        offset: Optional[tf.Transform3d] = None,
        geom_type: Optional[str] = None,
        geom_param: Any = None,
    ) -> None:
        if offset is None:
            self.offset: Optional[tf.Transform3d] = None
        else:
            self.offset = offset
        self.geom_type: Optional[str] = geom_type
        self.geom_param: Any = geom_param

    def __repr__(self) -> str:
        return f"Visual(offset={self.offset}, geom_type='{self.geom_type}', geom_param={self.geom_param})"


class Link:
    def __init__(
        self,
        name: Optional[str] = None,
        offset: Optional[tf.Transform3d] = None,
        visuals: Iterable[Visual] = (),
    ) -> None:
        if offset is None:
            self.offset: Optional[tf.Transform3d] = None
        else:
            self.offset = offset
        self.name: Optional[str] = name
        self.visuals: Iterable[Visual] = visuals

    def to(self, *args, **kwargs) -> "Link":
        if self.offset is not None:
            self.offset = self.offset.to(*args, **kwargs)
        return self

    def __repr__(self) -> str:
        return f"Link(name='{self.name}', offset={self.offset}, visuals={self.visuals})"


class Joint:
    TYPES: ClassVar = ["fixed", "revolute", "prismatic"]

    def __init__(
        self,
        name: Optional[str] = None,
        offset: Optional[tf.Transform3d] = None,
        joint_type: str = "fixed",
        axis: Optional[torch.Tensor] = (0.0, 0.0, 1.0),
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        limits: Optional[torch.Tensor] = None,
        velocity_limits: Optional[torch.Tensor] = None,
        effort_limits: Optional[torch.Tensor] = None,
    ) -> None:
        if offset is None:
            self.offset: Optional[tf.Transform3d] = None
        else:
            self.offset = offset
        self.name: Optional[str] = name
        if joint_type not in self.TYPES:
            raise RuntimeError(f"joint specified as {joint_type} type not, but we only support {self.TYPES}")
        self.joint_type: str = joint_type
        if axis is None:
            self.axis: torch.Tensor = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        elif torch.is_tensor(axis):
            self.axis = axis.clone().detach().to(dtype=dtype, device=device)
        else:
            self.axis = torch.tensor(axis, dtype=dtype, device=device)
        # normalize axis to have norm 1 (needed for correct representation scaling with theta)
        self.axis = self.axis / self.axis.norm()

        self.limits: Optional[torch.Tensor] = limits
        self.velocity_limits: Optional[torch.Tensor] = velocity_limits
        self.effort_limits: Optional[torch.Tensor] = effort_limits

    def to(self, *args, **kwargs) -> "Joint":
        self.axis = self.axis.to(*args, **kwargs)
        if self.offset is not None:
            self.offset = self.offset.to(*args, **kwargs)
        return self

    def clamp(self, joint_position: torch.Tensor) -> torch.Tensor:
        if self.limits is None:
            return joint_position
        else:
            return torch.clamp(joint_position, self.limits[0], self.limits[1])

    def __repr__(self) -> str:
        return f"Joint(name='{self.name}', offset={self.offset}, joint_type='{self.joint_type}', axis={self.axis})"


# prefix components:
space: str = "    "
branch: str = "│   "
# pointers:
tee: str = "├── "
last: str = "└── "


class Frame:
    def __init__(
        self,
        name: Optional[str] = None,
        link: Optional[Link] = None,
        joint: Optional[Joint] = None,
        children: Optional[List["Frame"]] = None,
    ) -> None:
        self.name: str = "None" if name is None else name
        self.link: Link = link if link is not None else Link()
        self.joint: Joint = joint if joint is not None else Joint()
        if children is None:
            self.children: List[Frame] = []
        else:
            self.children = children

    def __str__(self, prefix: str = "", root: bool = True) -> str:
        pointers = [tee] * (len(self.children) - 1) + [last]
        ret = prefix + self.name + "\n" if root else ""
        for pointer, child in zip(pointers, self.children):
            ret += prefix + pointer + child.name + "\n"
            if child.children:
                extension = branch if pointer == tee else space
                # i.e. space because last, └── , above so no more |
                ret += child.__str__(prefix=prefix + extension, root=False)
        return ret

    def to(self, *args, **kwargs) -> "Frame":
        self.joint = self.joint.to(*args, **kwargs)
        self.link = self.link.to(*args, **kwargs)
        self.children = [c.to(*args, **kwargs) for c in self.children]
        return self

    def add_child(self, child: "Frame") -> None:
        self.children.append(child)

    def is_end(self) -> bool:
        return len(self.children) == 0

    def get_transform(self, theta: torch.Tensor) -> tf.Transform3d:
        dtype = self.joint.axis.dtype
        d = self.joint.axis.device
        if self.joint.joint_type == "revolute":
            rot = axis_and_angle_to_matrix_33(self.joint.axis, theta)
            t = tf.Transform3d(rot=rot, dtype=dtype, device=d)
        elif self.joint.joint_type == "prismatic":
            pos = theta.unsqueeze(1) * self.joint.axis
            t = tf.Transform3d(pos=pos, dtype=dtype, device=d)
        elif self.joint.joint_type == "fixed":
            t = tf.Transform3d(
                default_batch_size=theta.shape[0],
                dtype=dtype,
                device=d,
            )
        else:
            raise ValueError(f"Unsupported joint type {self.joint.joint_type}.")
        if self.joint.offset is None:
            return t
        else:
            return self.joint.offset.compose(t)
