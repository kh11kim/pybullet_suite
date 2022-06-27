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
        ee_idx: Optional[int]=None,
    ):
        """Initialize Robot

        Args:
            physics_client (BulletClient): _description_
            body_uid (int): _description_
            ee_idx (Optional[int]): end effector frame of the robot. If none, the last link(frame) is used
        """
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid
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

    @contextmanager
    def no_set_joint(self, no_viz=False):
        if no_viz:
            self.physics_client.configureDebugVisualizer(
                self.physics_client.COV_ENABLE_RENDERING, 
                0
            )
        joints_temp = self.get_joint_angles()
        yield
        self.set_joint_angles(joints_temp)
        if no_viz:
            self.physics_client.configureDebugVisualizer(
                self.physics_client.COV_ENABLE_RENDERING, 
                1
            )
    
    @classmethod
    def make(
        cls,
        physics_client: BulletClient,
        pose: Pose = Pose.identity(),
        use_fixed_base: bool = True
    ):
        body_uid = physics_client.loadURDF(
            cls.urdf_path,
            pose.trans,
            pose.rot.as_quat(),
            useFixedBase=use_fixed_base,
        )
        return cls(physics_client, body_uid)



