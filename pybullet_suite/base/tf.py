import numpy as np
from typing import Union
import scipy.spatial.transform

class Rotation(scipy.spatial.transform.Rotation):
    """Wrapper of scipy.spatial.transform.Rotation class
    """
    @classmethod
    def identity(cls):
        return cls.from_quat([0.0, 0.0, 0.0, 1.0])


class Pose:
    """Rigid spatial transform between coordinate systems in 3D space.

    Attributes:
        rot (scipy.spatial.transform.Rotation)
        trans (np.ndarray)
    """

    def __init__(
        self, 
        rot: Rotation = Rotation.identity(), 
        trans: Union[np.ndarray, list] = [0.,0.,0.]):
        assert isinstance(rot, scipy.spatial.transform.Rotation)
        assert isinstance(trans, (np.ndarray, list, tuple))

        self.rot = rot
        self.trans = np.asarray(trans, np.double)
    
    def __eq__(self, other:"Pose"):
        same_rot = np.allclose(self.rot.as_quat(), other.rot.as_quat())
        same_trans = np.allclose(self.trans, other.trans)
        return same_rot & same_trans

    def as_matrix(self):
        """Represent as a 4x4 matrix."""
        return np.vstack(
            (np.c_[self.rot.as_matrix(), self.trans], [0.0, 0.0, 0.0, 1.0])
        )
    
    def as_1d_numpy(self):
        """Represet as a 1x7 vector [x,y,z,qx,qy,qz,qw]"""
        return np.hstack([*self.trans, *self.rot.as_quat()])

    # def to_dict(self):
    #     """Serialize Transform object into a dictionary."""
    #     return {
    #         "rotation": self.rot.as_quat().tolist(),
    #         "translation": self.trans.tolist(),
    #     }

    # def to_list(self):
    #     return np.r_[self.rot.as_quat(), self.trans]

    def __mul__(self, other: "Pose") -> "Pose":
        """Compose this transformation with another."""
        rotation = self.rot * other.rot
        translation = self.rot.apply(other.trans) + self.trans
        return self.__class__(rotation, translation)

    def transform_point(self, point: Union[np.ndarray, list]):
        return self.rot.apply(point) + self.trans

    def transform_vector(self, vector: Union[np.ndarray, list]):
        return self.rot.apply(vector)

    def inverse(self):
        """Compute the inverse of this transform."""
        rotation = self.rot.inv()
        translation = -rotation.apply(self.trans)
        return self.__class__(rotation, translation)

    @classmethod
    def from_matrix(cls, m):
        """Initialize from a 4x4 matrix."""
        rotation = Rotation.from_matrix(m[:3, :3])
        translation = m[:3, 3]
        return cls(rotation, translation)

    @classmethod
    def identity(cls):
        """Initialize with the identity transformation."""
        rotation = Rotation.from_quat([0.0, 0.0, 0.0, 1.0])
        translation = np.array([0.0, 0.0, 0.0])
        return cls(rotation, translation)

    @classmethod
    def look_at(cls, eye, center, up):
        """Initialize with a LookAt matrix.

        Returns:
            T_eye_ref, the transform from camera to the reference frame, w.r.t.
            which the input arguments were defined.
        """
        eye = np.asarray(eye)
        center = np.asarray(center)

        forward = center - eye
        forward /= np.linalg.norm(forward)

        right = np.cross(forward, up)
        right /= np.linalg.norm(right)

        up = np.asarray(up) / np.linalg.norm(up)
        up = np.cross(right, forward)

        m = np.eye(4, 4)
        m[:3, 0] = right
        m[:3, 1] = -up
        m[:3, 2] = forward
        m[:3, 3] = eye

        return cls.from_matrix(m).inverse()

# Quaternion functions
def qtn_conj(qtn):
    return np.hstack([-qtn[:3], qtn[-1]])

def qtn_mul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return np.array([x, y, z, w])

def orn_error(desired, current):
    cc = qtn_conj(current)
    q_r = qtn_mul(desired, cc)
    return q_r[:3] * np.sign(q_r[-1])

def slerp(qtn1, qtn2, ratio):
    if np.allclose(qtn1, qtn2) | np.allclose(qtn1, -qtn2):
        return qtn1
    if qtn1 @ qtn2 < 0.:
        qtn2 = -qtn2
    theta = np.arccos(qtn1@qtn2)
    ratio_ = 1. - ratio
    wa = np.sin(ratio_*theta) / np.sin(theta)
    wb = np.sin(ratio*theta) / np.sin(theta)
    qtn = wa*qtn1 + wb*qtn2
    return qtn/np.linalg.norm(qtn)