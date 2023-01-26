import numpy as np
from typing import Optional, Union
from contextlib import contextmanager

from pybullet_utils.bullet_client import BulletClient
from .body import Body
from .tf import Rotation, Pose


class Robot(Body):
    """Actuated body(robot) class.
    """

    def __init__(
        self,
        physics_client: BulletClient,
        body_uid: int,
        name: str,
        ee_idx: Optional[int] = None,
    ):
        """Initialize Robot

        Args:
            physics_client (BulletClient): _description_
            body_uid (int): _description_
            ee_idx (Optional[int]): end effector frame of the robot. If none, the last link(frame) is used
        """
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid,
            name=name
        )
        self.ee_idx = ee_idx if ee_idx is not None else (self.n_joints - 1)
        self.set_joint_limits()
        self.set_joint_angles(self.joint_central)

    def set_joint_limits(self):
        ll_list = []
        ul_list = []
        mid_list = []
        for joint in self.info:
            ll = self.info[joint]["joint_lower_limit"]
            ul = self.info[joint]["joint_upper_limit"]
            mid = (ll + ul) / 2
            ll_list.append(ll)
            ul_list.append(ul)
            mid_list.append(mid)
        self.joint_lower_limit = np.asarray(ll_list)
        self.joint_upper_limit = np.asarray(ul_list)
        self.joint_central = np.asarray(mid_list)

    def get_ee_pose(self):
        return self.get_link_pose(self.ee_idx)

    def inverse_kinematics(
        self,
        pos: np.ndarray = None,
        pose: Pose = None,
        tol: float = 1e-3,
        max_iter: int = 100,
        start_central=True
    ):
        def is_pose_close(pose1: Pose, pose2: Pose, tol: float):
            if np.linalg.norm(pose1.trans - pose2.trans) > tol:
                return False
            if pose1.rot.angle_between(pose2.rot) > tol*np.pi:
                return False
            return True
        def is_pos_close(pose1, pos, tol):
            if np.linalg.norm(pose1.trans - pos) > tol:
                return False
            return True

        assert (pos is None) or (pose is None)
        orn = None
        success = False
        if pose is not None:
            pos, orn = pose.trans, pose.rot.as_quat()
        with self.no_set_joint():
            if start_central:
                self.set_joint_angles(self.arm_central)
            for _ in range(max_iter):
                joint_angles = self.physics_client.calculateInverseKinematics(
                    bodyIndex=self.uid,
                    endEffectorLinkIndex=self.ee_idx,
                    targetPosition=pos,
                    targetOrientation=orn
                )
                self.set_joint_angles(joint_angles)
                pose_curr = self.get_ee_pose()
                if pos is None:
                    if is_pose_close(pose_curr, pose, tol):
                        success = True
                        break
                else:
                    if is_pos_close(pose_curr, pos, tol):
                        success = True
                        break
        if success:
            return np.array(joint_angles)
        return None

    def forward_kinematics(
        self,
        angles: np.ndarray
    ):
        with self.no_set_joint():
            self.set_joint_angles(angles)
            pose = self.get_ee_pose()
        return pose

    def get_jacobian(
        self,
        link_idx=None,
        joint_angles: Optional[np.ndarray] = None,
        local_position: Union[list, np.ndarray] = [0, 0, 0]
    ):
        if link_idx is None:
            link_idx = self.ee_idx
        if joint_angles is None:
            joint_angles = self.get_joint_angles()
        jac_trans, jac_rot = self.physics_client.calculateJacobian(
            bodyUniqueId=self.uid,
            linkIndex=link_idx,
            localPosition=local_position,
            objPositions=joint_angles.tolist(),
            objVelocities=np.zeros_like(joint_angles).tolist(),
            objAccelerations=np.zeros_like(joint_angles).tolist()
        )
        return np.vstack([jac_trans, jac_rot])
    
    

    @contextmanager
    def no_set_joint(self):
        joints_temp = self.get_joint_angles()
        yield
        self.set_joint_angles(joints_temp)

    @classmethod
    def make(
        cls,
        physics_client: BulletClient,
        name: str,
        pose: Pose = Pose.identity(),
        use_fixed_base: bool = True,
        
    ):
        body_uid = physics_client.loadURDF(
            cls.urdf_path,
            pose.trans,
            pose.rot.as_quat(),
            useFixedBase=use_fixed_base,
        )
        return cls(physics_client, body_uid, name)
