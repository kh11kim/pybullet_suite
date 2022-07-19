from ..base import *
from ...data import PANDA_URDF

class Panda(Robot):
    urdf_path = PANDA_URDF
    def __init__(self, physics_client: BulletClient, body_uid: int):
        self.arm_idxs = range(7)
        self.finger_idxs = [8, 9]
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid,
            ee_idx = 10
        )
        self.max_opening_width = 0.08
        self.arm_lower_limit = self.joint_lower_limit[self.arm_idxs]
        self.arm_upper_limit = self.joint_upper_limit[self.arm_idxs]
        self.arm_central = (self.arm_lower_limit + self.arm_upper_limit)/2
        self.open()

    def inverse_kinematics(
        self, 
        pos: np.ndarray = None, 
        pose: Pose = None,
        tol: float = 5e-3,
        max_iter: int = 10
    ):
        assert (pos is None) ^ (pose is None)
        orn = None
        success = False
        if pose is not None:
            pos, orn = pose.trans, pose.rot.as_quat()
        with self.no_set_joint():
            for i in range(max_iter):
                joint_angles = self.physics_client.calculateInverseKinematics(
                    bodyIndex=self.uid,
                    endEffectorLinkIndex=self.ee_idx,
                    targetPosition=pos,
                    targetOrientation=orn
                )
                self.set_joint_angles(joint_angles)
                pose_curr = self.get_ee_pose()
                if np.linalg.norm(pose_curr.trans - pos) < tol:
                    success = True
                    break
        if success:
            return np.array(joint_angles)[self.arm_idxs]
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
        joint_angles: Optional[np.ndarray] = None,
        local_position: Union[list, np.ndarray] = [0,0,0]
    ):
        if joint_angles is not None:
            assert len(self.arm_idxs) == len(joint_angles)
            joint_angles = np.asarray([*joint_angles, 0, 0])
        if joint_angles is None:
            joint_angles = self.get_joint_angles()
        jac_trans, jac_rot = self.physics_client.calculateJacobian(
            bodyUniqueId=self.uid,
            linkIndex=self.ee_idx,
            localPosition=local_position,
            objPositions=joint_angles.tolist(),
            objVelocities=np.zeros_like(joint_angles).tolist(),
            objAccelerations=np.zeros_like(joint_angles).tolist()
        )
        return np.vstack([jac_trans, jac_rot])[:,:-2]

    def get_joint_angles(self):
        return super().get_joint_angles()[self.arm_idxs]
    
    def get_random_arm_angles(self):
        q = np.random.uniform(low=self.arm_lower_limit, high=self.arm_upper_limit)
        return q

    def set_joint_angles(self, angles: np.ndarray):
        for i, angle in zip(self.arm_idxs, angles):
            super().set_joint_angle(joint=i, angle=angle) 
    
    def open(self, ctrl=False):
        if ctrl == False:
            for i in self.finger_idxs:
                self.set_joint_angle(joint=i, angle=self.max_opening_width/2)
    
    def close(self, ctrl=False):
        if ctrl == False:
            for i in self.finger_idxs:
                self.set_joint_angle(joint=i, angle=0)




if __name__ == "__main__":
    world = BulletWorld(gui=True)
    robot: Panda = world.load_robot("robot", robot_class=Panda)
    robot.set_joint_angles([0,0,0,0,0,0,0])
    input()