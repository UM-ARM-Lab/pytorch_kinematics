# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import math
import typing
import warnings
from typing import Optional

import torch

from .rotation_conversions import _axis_angle_rotation, matrix_to_quaternion, quaternion_to_matrix, \
    euler_angles_to_matrix
from pytorch_kinematics.transforms.perturbation import sample_perturbations

DEFAULT_EULER_CONVENTION = "XYZ"


class Transform3d:
    """
    A Transform3d object encapsulates a batch of N 3D transformations, and knows
    how to transform points and normal vectors. Suppose that t is a Transform3d;
    then we can do the following:

    .. code-block:: python

        N = len(t)
        points = torch.randn(N, P, 3)
        normals = torch.randn(N, P, 3)
        points_transformed = t.transform_points(points)    # => (N, P, 3)
        normals_transformed = t.transform_normals(normals)  # => (N, P, 3)


    BROADCASTING
    Transform3d objects supports broadcasting. Suppose that t1 and tN are
    Transform3D objects with len(t1) == 1 and len(tN) == N respectively. Then we
    can broadcast transforms like this:

    .. code-block:: python

        t1.transform_points(torch.randn(P, 3))     # => (P, 3)
        t1.transform_points(torch.randn(1, P, 3))  # => (1, P, 3)
        t1.transform_points(torch.randn(M, P, 3))  # => (M, P, 3)
        tN.transform_points(torch.randn(P, 3))     # => (N, P, 3)
        tN.transform_points(torch.randn(1, P, 3))  # => (N, P, 3)


    COMBINING TRANSFORMS
    Transform3d objects can be combined in two ways: composing and stacking.
    Composing is function composition. Given Transform3d objects t1, t2, t3,
    the following all compute the same thing:

    .. code-block:: python

        y1 = t3.transform_points(t2.transform_points(t1.transform_points(x)))
        y2 = t1.compose(t2).compose(t3).transform_points(x)
        y3 = t1.compose(t2, t3).transform_points(x)


    Composing transforms should broadcast.

    .. code-block:: python

        if len(t1) == 1 and len(t2) == N, then len(t1.compose(t2)) == N.

    We can also stack a sequence of Transform3d objects, which represents
    composition along the batch dimension; then the following should compute the
    same thing.

    .. code-block:: python

        N, M = len(tN), len(tM)
        xN = torch.randn(N, P, 3)
        xM = torch.randn(M, P, 3)
        y1 = torch.cat([tN.transform_points(xN), tM.transform_points(xM)], dim=0)
        y2 = tN.stack(tM).transform_points(torch.cat([xN, xM], dim=0))

    BUILDING TRANSFORMS
    We provide convenience methods for easily building Transform3d objects
    as compositions of basic transforms.

    .. code-block:: python

        # Scale by 0.5, then translate by (1, 2, 3)
        t1 = Transform3d().scale(0.5).translate(1, 2, 3)

        # Scale each axis by a different amount, then translate, then scale
        t2 = Transform3d().scale(1, 3, 3).translate(2, 3, 1).scale(2.0)

        t3 = t1.compose(t2)
        tN = t1.stack(t3, t3)


    BACKPROP THROUGH TRANSFORMS
    When building transforms, we can also parameterize them by Torch tensors;
    in this case we can backprop through the construction and application of
    Transform objects, so they could be learned via gradient descent or
    predicted by a neural network.

    .. code-block:: python

        s1_params = torch.randn(N, requires_grad=True)
        t_params = torch.randn(N, 3, requires_grad=True)
        s2_params = torch.randn(N, 3, requires_grad=True)

        t = Transform3d().scale(s1_params).translate(t_params).scale(s2_params)
        x = torch.randn(N, 3)
        y = t.transform_points(x)
        loss = compute_loss(y)
        loss.backward()

        with torch.no_grad():
            s1_params -= lr * s1_params.grad
            t_params -= lr * t_params.grad
            s2_params -= lr * s2_params.grad

    CONVENTIONS
    We adopt a right-hand coordinate system, meaning that rotation about an axis
    with a positive angle results in a counter clockwise rotation.

    This class assumes that transformations are applied on inputs which
    are column vectors (different from pytorch3d!). The internal representation of the Nx4x4 transformation
    matrix is of the form:

    .. code-block:: python

        M = [
                [Rxx, Ryx, Rzx, Tx],
                [Rxy, Ryy, Rzy, Ty],
                [Rxz, Ryz, Rzz, Tz],
                [0,  0,  0,  1],
            ]

    To apply the transformation to points which are row vectors, the M matrix
    can be pre multiplied by the points:

    .. code-block:: python

        points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
        transformed_point = M @ points[0]
        transformed_points = points @ M.transpose(-1,-2)

    Euler angles given as input by default are interpreted to be in "RXYZ" convention.
    Quaternions given as input should be in [w,x,y,z] order.
    """

    def __init__(
            self,
            default_batch_size=1,
            dtype: torch.dtype = torch.float32,
            device='cpu',
            matrix: Optional[torch.Tensor] = None,
            rot: Optional[typing.Iterable] = None,
            pos: Optional[typing.Iterable] = None,
    ):
        """
        Args:
            default_batch_size: A positive integer representing the minibatch size
                if matrix is None, rot is None, and pos is also None.
            dtype: The data type of the transformation matrix.
                to be used if `matrix = None`.
            device: The device for storing the implemented transformation.
                If `matrix != None`, uses the device of input `matrix`.
            matrix: A tensor of shape (4, 4) or of shape (minibatch, 4, 4)
                representing the 4x4 3D transformation matrix.
                If `None`, initializes with identity using
                the specified `device` and `dtype`.
            rot: A rotation matrix of shape (3, 3) or of shape (minibatch, 3, 3), or
                a quaternion of shape (4,) or of shape (minibatch, 4), where
                minibatch should match that of matrix if that is also passed in.
                The rotation overrides the rotation given in the matrix argument, if any.
                Quaternions must be in wxyz order.
            pos: A tensor of shape (3,) or of shape (minibatch, 3) representing the position
                offsets of the transforms, where minibatch should match that of matrix if
                that is also passed in. The position overrides the position given in the
                matrix argument, if any.
        """
        if matrix is None:
            self._matrix = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).repeat(default_batch_size, 1, 1)
        else:
            if matrix.ndim not in (2, 3):
                raise ValueError('"matrix" has to be a 2- or a 3-dimensional tensor.')
            if matrix.shape[-2] != 4 or matrix.shape[-1] != 4:
                raise ValueError(
                    '"matrix" has to be a tensor of shape (minibatch, 4, 4)'
                )
            # set the device from matrix
            device = matrix.device
            self._matrix = matrix.view(-1, 4, 4)

        if pos is not None:
            ones = torch.ones([1], dtype=dtype, device=device)
            if not torch.is_tensor(pos):
                pos = torch.tensor(pos, dtype=dtype, device=device)
            if pos.ndim in (2, 3):
                ones = ones.repeat(*pos.shape[:-1], 1)
                if pos.ndim in (2, 3) and pos.shape[0] > 1 and self._matrix.shape[0] == 1:
                    self._matrix = self._matrix.repeat(pos.shape[0], 1, 1)
            pos_h = torch.cat((pos, ones), dim=-1).reshape(-1, 4, 1)
            self._matrix = torch.cat((self._matrix[:, :, :3], pos_h), dim=-1)

        if rot is not None:
            zeros = torch.zeros(1, 3, dtype=dtype, device=device)
            if not torch.is_tensor(rot):
                rot = torch.tensor(rot, dtype=dtype, device=device)
            if rot.shape[-1] == 4:
                rot = quaternion_to_matrix(rot)
            elif rot.shape[-1] == 3 and (len(rot.shape) == 1 or rot.shape[-2] != 3):
                rot = euler_angles_to_matrix(rot, DEFAULT_EULER_CONVENTION)
            if rot.ndim == 3:
                zeros = zeros.repeat(rot.shape[0], 1, 1)
                if rot.shape[0] > 1 and self._matrix.shape[0] == 1:
                    self._matrix = self._matrix.repeat(rot.shape[0], 1, 1)
            rot_h = torch.cat((rot, zeros), dim=-2).reshape(-1, 4, 3)
            self._matrix = torch.cat((rot_h, self._matrix[:, :, 3].reshape(-1, 4, 1)), dim=-1)

        self._lu = None
        self.device = device
        self.dtype = self._matrix.dtype

    def __len__(self):
        return self.get_matrix().shape[0]

    def __getitem__(self, item):
        return Transform3d(matrix=self.get_matrix()[item])

    def __repr__(self):
        m = self.get_matrix()
        pos = m[:, :3, 3]
        rot = matrix_to_quaternion(m[:, :3, :3])
        return "Transform3d(rot={}, pos={})".format(rot, pos).replace('\n       ', '')

    def compose(self, *others):
        """
        Return a new Transform3d with the tranforms to compose stored as
        an internal list.

        Args:
            *others: Any number of Transform3d objects

        Returns:
            A new Transform3d with the stored transforms
        """

        mat = self._matrix
        for other in others:
            mat = _broadcast_bmm(mat, other.get_matrix())

        out = Transform3d(device=self.device, dtype=self.dtype, matrix=mat)
        return out

    def get_matrix(self):
        """
        Return the Nx4x4 homogeneous transformation matrix represented by this object.
        """
        return self._matrix

    def _get_matrix_inverse(self):
        """
        Return the inverse of self._matrix.
        """

        return self._invert_transformation_matrix(self._matrix)

    @staticmethod
    def _invert_transformation_matrix(T):
        """
        Invert homogeneous transformation matrix.
        """
        Tinv = T.clone()
        R = T[:, :3, :3]
        t = T[:, :3, 3]
        Tinv[:, :3, :3] = R.transpose(1, 2)
        Tinv[:, :3, 3:] = -Tinv[:, :3, :3] @ t.unsqueeze(-1)
        return Tinv

    def inverse(self, invert_composed: bool = False):
        """
        Returns a new Transform3D object that represents an inverse of the
        current transformation.

        Args:
            invert_composed: ignored, included for backwards compatibility

        Returns:
            A new Transform3D object containing the inverse of the original
            transformation.
        """

        i_matrix = self._get_matrix_inverse()

        tinv = Transform3d(matrix=i_matrix, device=self.device)

        return tinv

    def stack(self, *others):
        transforms = [self] + list(others)
        matrix = torch.cat([t._matrix for t in transforms], dim=0).to(self.device, self.dtype)
        out = self.__class__(device=self.device, dtype=self.dtype)
        out._matrix = matrix
        return out

    def transform_points(self, points, eps: Optional[float] = None):
        """
        Use this transform to transform a set of 3D points. Assumes row major
        ordering of the input points.

        Args:
            points: Tensor of shape (P, 3) or (N, P, 3)
            eps: If eps!=None, the argument is used to clamp the
                last coordinate before peforming the final division.
                The clamping corresponds to:
                last_coord := (last_coord.sign() + (last_coord==0)) *
                torch.clamp(last_coord.abs(), eps),
                i.e. the last coordinates that are exactly 0 will
                be clamped to +eps.

        Returns:
            points_out: points of shape (N, P, 3) or (P, 3) depending
            on the dimensions of the transform
        """
        points_batch = points.clone()
        if points_batch.dim() == 2:
            points_batch = points_batch[None]  # (P, 3) -> (1, P, 3)
        if points_batch.dim() != 3:
            msg = "Expected points to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % repr(points.shape))

        N, P, _3 = points_batch.shape
        ones = torch.ones(N, P, 1, dtype=points.dtype, device=points.device)
        points_batch = torch.cat([points_batch, ones], dim=2)

        composed_matrix = self.get_matrix().transpose(-1, -2)
        points_out = _broadcast_bmm(points_batch, composed_matrix)
        denom = points_out[..., 3:]  # denominator
        if eps is not None:
            denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
            denom = denom_sign * torch.clamp(denom.abs(), eps)
        points_out = points_out[..., :3] / denom

        # When transform is (1, 4, 4) and points is (P, 3) return
        # points_out of shape (P, 3)
        if points_out.shape[0] == 1 and points.dim() == 2:
            points_out = points_out.reshape(points.shape)

        return points_out

    def transform_normals(self, normals):
        """
        Use this transform to transform a set of normal vectors.

        Args:
            normals: Tensor of shape (P, 3) or (N, P, 3)

        Returns:
            normals_out: Tensor of shape (P, 3) or (N, P, 3) depending
            on the dimensions of the transform
        """
        if normals.dim() not in [2, 3]:
            msg = "Expected normals to have dim = 2 or dim = 3: got shape %r"
            raise ValueError(msg % (normals.shape,))
        mat = self.inverse().get_matrix()[:, :3, :3]
        normals_out = _broadcast_bmm(normals, mat)

        # This doesn't pass unit tests. TODO investigate further
        # if self._lu is None:
        #     self._lu = self._matrix[:, :3, :3].transpose(1, 2).lu()
        # normals_out = normals.lu_solve(*self._lu)

        # When transform is (1, 4, 4) and normals is (P, 3) return
        # normals_out of shape (P, 3)
        if normals_out.shape[0] == 1 and normals.dim() == 2:
            normals_out = normals_out.reshape(normals.shape)

        return normals_out

    def transform_shape_operator(self, shape_operators):
        """
        Use this transform to transform a set of shape_operator (or Weingarten map).
        This is the hessian of a signed-distance, i.e. gradient of a normal vector.

        Args:
            shape_operators: Tensor of shape (P, 3, 3) or (N, P, 3, 3)

        Returns:
            shape_operators_out: Tensor of shape (P, 3, 3) or (N, P, 3, 3) depending
            on the dimensions of the transform
        """
        if shape_operators.dim() not in [3, 4]:
            msg = "Expected shape_operators to have dim = 3 or dim = 4: got shape %r"
            raise ValueError(msg % (shape_operators.shape,))
        mat = self.inverse().get_matrix()[:, :3, :3]
        shape_operators_out = _broadcast_bmm(mat.permute(0, 2, 1), _broadcast_bmm(shape_operators, mat))

        # When transform is (1, 4, 4) and shape_operator is (P, 3, 3) return
        # shape_operators_out of shape (P, 3, 3)
        if shape_operators_out.shape[0] == 1 and shape_operators.dim() == 3:
            shape_operators_out = shape_operators_out.reshape(shape_operators.shape)

        return shape_operators_out

    def translate(self, *args, **kwargs):
        return self.compose(Translate(device=self.device, *args, **kwargs))

    def scale(self, *args, **kwargs):
        return self.compose(Scale(device=self.device, *args, **kwargs))

    def rotate(self, *args, **kwargs):
        return self.compose(Rotate(device=self.device, *args, **kwargs))

    def rotate_axis_angle(self, *args, **kwargs):
        return self.compose(RotateAxisAngle(device=self.device, *args, **kwargs))

    def sample_perturbations(self, num_perturbations, radian_sigma, translation_sigma):
        mat = self.get_matrix()
        if mat.shape[0] == 1:
            mat = mat[0]
        all_mats = sample_perturbations(mat, num_perturbations, radian_sigma, translation_sigma)
        out = Transform3d(matrix=all_mats)
        return out

    def clone(self):
        """
        Deep copy of Transforms object. All internal tensors are cloned
        individually.

        Returns:
            new Transforms object.
        """
        other = Transform3d(dtype=self.dtype, device=self.device)
        if self._lu is not None:
            other._lu = [elem.clone() for elem in self._lu]
        other._matrix = self._matrix.clone()
        return other

    def to(self, device, copy: bool = False, dtype=None):
        """
        Match functionality of torch.Tensor.to()
        If copy = True or the self Tensor is on a different device, the
        returned tensor is a copy of self with the desired torch.device.
        If copy = False and the self Tensor already has the correct torch.device,
        then self is returned.

        Args:
          device: Device id for the new tensor.
          copy: Boolean indicator whether or not to clone self. Default False.
          dtype: If not None, casts the internal tensor variables
              to a given torch.dtype.

        Returns:
          Transform3d object.
        """
        if not copy and (dtype is None or self.dtype == dtype) and self.device == device:
            return self
        other = self.clone()
        other.device = device
        other.dtype = dtype if dtype is not None else other.dtype
        other._matrix = self._matrix.to(device=device, dtype=dtype)
        return other

    def cpu(self):
        return self.to(torch.device("cpu"))

    def cuda(self):
        return self.to(torch.device("cuda"))


class Translate(Transform3d):
    def __init__(self, x, y=None, z=None, dtype=torch.float32, device: str = "cpu"):
        """
        Create a new Transform3d representing 3D translations.

        Option I: Translate(xyz, dtype=torch.float32, device='cpu')
            xyz should be a tensor of shape (N, 3)

        Option II: Translate(x, y, z, dtype=torch.float32, device='cpu')
            Here x, y, and z will be broadcast against each other and
            concatenated to form the translation. Each can be:
                - A python scalar
                - A torch scalar
                - A 1D torch tensor
        """
        super().__init__(device=device)
        xyz = _handle_input(x, y, z, dtype, device, "Translate")
        N = xyz.shape[0]

        mat = torch.eye(4, dtype=dtype, device=device)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, :3, 3] = xyz
        self._matrix = mat

    def _get_matrix_inverse(self):
        """
        Return the inverse of self._matrix.
        """
        inv_mask = self._matrix.new_ones([1, 4, 4])
        inv_mask[0, :3, 3] = -1.0
        i_matrix = self._matrix * inv_mask
        return i_matrix


class Scale(Transform3d):
    def __init__(self, x, y=None, z=None, dtype=torch.float32, device: str = "cpu"):
        """
        A Transform3d representing a scaling operation, with different scale
        factors along each coordinate axis.

        Option I: Scale(s, dtype=torch.float32, device='cpu')
            s can be one of
                - Python scalar or torch scalar: Single uniform scale
                - 1D torch tensor of shape (N,): A batch of uniform scale
                - 2D torch tensor of shape (N, 3): Scale differently along each axis

        Option II: Scale(x, y, z, dtype=torch.float32, device='cpu')
            Each of x, y, and z can be one of
                - python scalar
                - torch scalar
                - 1D torch tensor
        """
        super().__init__(device=device)
        xyz = _handle_input(x, y, z, dtype, device, "scale", allow_singleton=True)
        N = xyz.shape[0]

        # TODO: Can we do this all in one go somehow?
        mat = torch.eye(4, dtype=dtype, device=device)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, 0, 0] = xyz[:, 0]
        mat[:, 1, 1] = xyz[:, 1]
        mat[:, 2, 2] = xyz[:, 2]
        self._matrix = mat

    def _get_matrix_inverse(self):
        """
        Return the inverse of self._matrix.
        """
        xyz = torch.stack([self._matrix[:, i, i] for i in range(4)], dim=1)
        ixyz = 1.0 / xyz
        imat = torch.diag_embed(ixyz, dim1=1, dim2=2)
        return imat


class Rotate(Transform3d):
    def __init__(
            self, R, dtype=torch.float32, device: str = "cpu", orthogonal_tol: float = 1e-5
    ):
        """
        Create a new Transform3d representing 3D rotation using a rotation
        matrix as the input.

        Args:
            R: a tensor of shape (3, 3) or (N, 3, 3)
            orthogonal_tol: tolerance for the test of the orthogonality of R

        """
        super().__init__(device=device)
        if not torch.is_tensor(R):
            R = torch.tensor(R, dtype=dtype, device=device)
        R = R.to(dtype=dtype).to(device=device)
        if R.shape[-1] == 4:
            R = quaternion_to_matrix(R)
        elif R.shape[-1] == 3 and (len(R.shape) == 1 or R.shape[-2] != 3):
            R = euler_angles_to_matrix(R, DEFAULT_EULER_CONVENTION)
        else:
            _check_valid_rotation_matrix(R.view(-1, 3, 3), tol=orthogonal_tol)
        if R.dim() == 2:
            R = R[None]

        if R.shape[-2:] != (3, 3):
            msg = "R must have shape (3, 3) or (N, 3, 3); got %s"
            raise ValueError(msg % repr(R.shape))
        N = R.shape[0]
        mat = torch.eye(4, dtype=dtype, device=device)
        mat = mat.view(1, 4, 4).repeat(N, 1, 1)
        mat[:, :3, :3] = R
        self._matrix = mat

    def _get_matrix_inverse(self):
        """
        Return the inverse of self._matrix.
        """
        return self._matrix.permute(0, 2, 1).contiguous()


class RotateAxisAngle(Rotate):
    def __init__(
            self,
            angle,
            axis: str = "X",
            degrees: bool = True,
            dtype=torch.float64,
            device: str = "cpu",
    ):
        """
        Create a new Transform3d representing 3D rotation about an axis
        by an angle.

        Assuming a right-hand coordinate system, positive rotation angles result
        in a counter clockwise rotation.

        Args:
            angle:
                - A torch tensor of shape (N,)
                - A python scalar
                - A torch scalar
            axis:
                string: one of ["X", "Y", "Z"] indicating the axis about which
                to rotate.
                NOTE: All batch elements are rotated about the same axis.
        """
        axis = axis.upper()
        if axis not in ["X", "Y", "Z"]:
            msg = "Expected axis to be one of ['X', 'Y', 'Z']; got %s"
            raise ValueError(msg % axis)
        angle = _handle_angle_input(angle, dtype, device, "RotateAxisAngle")
        angle = (angle / 180.0 * math.pi) if degrees else angle
        # We assume the points on which this transformation will be applied
        # are row vectors. The rotation matrix returned from _axis_angle_rotation
        # is for transforming column vectors. Therefore we transpose this matrix.
        # R will always be of shape (N, 3, 3)
        R = _axis_angle_rotation(axis, angle)
        super().__init__(device=device, R=R)


def _handle_coord(c, dtype, device):
    """
    Helper function for _handle_input.

    Args:
        c: Python scalar, torch scalar, or 1D torch tensor

    Returns:
        c_vec: 1D torch tensor
    """
    if not torch.is_tensor(c):
        c = torch.tensor(c, dtype=dtype, device=device)
    if c.dim() == 0:
        c = c.view(1)
    return c


def _handle_input(x, y, z, dtype, device, name: str, allow_singleton: bool = False):
    """
    Helper function to handle parsing logic for building transforms. The output
    is always a tensor of shape (N, 3), but there are several types of allowed
    input.

    Case I: Single Matrix
        In this case x is a tensor of shape (N, 3), and y and z are None. Here just
        return x.

    Case II: Vectors and Scalars
        In this case each of x, y, and z can be one of the following
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        In this case x, y and z are broadcast to tensors of shape (N, 1)
        and concatenated to a tensor of shape (N, 3)

    Case III: Singleton (only if allow_singleton=True)
        In this case y and z are None, and x can be one of the following:
            - Python scalar
            - Torch scalar
            - Torch tensor of shape (N, 1) or (1, 1)
        Here x will be duplicated 3 times, and we return a tensor of shape (N, 3)

    Returns:
        xyz: Tensor of shape (N, 3)
    """
    # If x is actually a tensor of shape (N, 3) then just return it
    if torch.is_tensor(x) and x.dim() == 2:
        if x.shape[1] != 3:
            msg = "Expected tensor of shape (N, 3); got %r (in %s)"
            raise ValueError(msg % (x.shape, name))
        if y is not None or z is not None:
            msg = "Expected y and z to be None (in %s)" % name
            raise ValueError(msg)
        return x

    if allow_singleton and y is None and z is None:
        y = x
        z = x

    # Convert all to 1D tensors
    xyz = [_handle_coord(c, dtype, device) for c in [x, y, z]]

    # Broadcast and concatenate
    sizes = [c.shape[0] for c in xyz]
    N = max(sizes)
    for c in xyz:
        if c.shape[0] != 1 and c.shape[0] != N:
            msg = "Got non-broadcastable sizes %r (in %s)" % (sizes, name)
            raise ValueError(msg)
    xyz = [c.expand(N) for c in xyz]
    xyz = torch.stack(xyz, dim=1)
    return xyz


def _handle_angle_input(x, dtype, device: str, name: str):
    """
    Helper function for building a rotation function using angles.
    The output is always of shape (N,).

    The input can be one of:
        - Torch tensor of shape (N,)
        - Python scalar
        - Torch scalar
    """
    if torch.is_tensor(x) and x.dim() > 1:
        msg = "Expected tensor of shape (N,); got %r (in %s)"
        raise ValueError(msg % (x.shape, name))
    else:
        return _handle_coord(x, dtype, device)


def _broadcast_bmm(a, b):
    """
    Batch multiply two matrices and broadcast if necessary.

    Args:
        a: torch tensor of shape (P, K) or (M, P, K)
        b: torch tensor of shape (N, K, K)

    Returns:
        a and b broadcast multipled. The output batch dimension is max(N, M).

    To broadcast transforms across a batch dimension if M != N then
    expect that either M = 1 or N = 1. The tensor with batch dimension 1 is
    expanded to have shape N or M.
    """
    if a.dim() == 2:
        a = a[None]
    if len(a) != len(b):
        if not ((len(a) == 1) or (len(b) == 1)):
            msg = "Expected batch dim for bmm to be equal or 1; got %r, %r"
            raise ValueError(msg % (a.shape, b.shape))
        if len(a) == 1:
            a = a.expand(len(b), -1, -1)
        if len(b) == 1:
            b = b.expand(len(a), -1, -1)
    return a.bmm(b)


def _check_valid_rotation_matrix(R, tol: float = 1e-7):
    """
    Determine if R is a valid rotation matrix by checking it satisfies the
    following conditions:

    ``RR^T = I and det(R) = 1``

    Args:
        R: an (N, 3, 3) matrix

    Returns:
        None

    Emits a warning if R is an invalid rotation matrix.
    """
    N = R.shape[0]
    eye = torch.eye(3, dtype=R.dtype, device=R.device)
    eye = eye.view(1, 3, 3).expand(N, -1, -1)
    orthogonal = torch.allclose(R.bmm(R.transpose(1, 2)), eye, atol=tol)
    det_R = torch.det(R)
    no_distortion = torch.allclose(det_R, torch.ones_like(det_R))
    if not (orthogonal and no_distortion):
        msg = "R is not a valid rotation matrix"
        warnings.warn(msg)
    return
