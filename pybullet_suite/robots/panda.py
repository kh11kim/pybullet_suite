from ..base import *
from ..utils.utils import PANDA_URDF

class Panda(Robot):
    urdf_path = PANDA_URDF.as_posix()

    def __init__(self, physics_client: BulletClient, body_uid: int):
        self.arm_idxs = range(7)
        self.finger_idxs = [8, 9]
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid,
            ee_idx=10
        )
        self.max_opening_width = 0.08
        self.arm_lower_limit = self.joint_lower_limit[self.arm_idxs]
        self.arm_upper_limit = self.joint_upper_limit[self.arm_idxs]
        self.arm_central = (self.arm_lower_limit + self.arm_upper_limit)/2
        self.open()

    def get_joint_angles(self):
        return super().get_joint_angles()[self.arm_idxs]

    def get_random_arm_angles(self):
        q = np.random.uniform(low=self.arm_lower_limit,
                              high=self.arm_upper_limit)
        return q

    def set_joint_angles(self, angles: np.ndarray):
        for i, angle in zip(self.arm_idxs, angles):
            super().set_joint_angle(joint=i, angle=angle)

    def open(self, width: float = None, ctrl=False):
        if width is None:
            angle = self.max_opening_width/2
        else:
            angle = width/2
        if ctrl == False:
            for i in self.finger_idxs:
                self.set_joint_angle(joint=i, angle=angle)

    def close(self, ctrl=False):
        if ctrl == False:
            for i in self.finger_idxs:
                self.set_joint_angle(joint=i, angle=0)

    def inverse_kinematics(
        self,
        pos: np.ndarray = None,
        pose: Pose = None,
        tol: float = 1e-3,
        max_iter: int = 100,
        start_central=True
    ):
        return super().inverse_kinematics(
            pos, pose, tol, max_iter, start_central
        )[self.arm_idxs]

    # forward kinematics do not need overriding

    def get_jacobian(
        self,
        joint_angles: Optional[np.ndarray] = None,
        link_idx=None,
        local_position: Union[list, np.ndarray] = [0, 0, 0]
    ):
        if link_idx is None:
            link_idx = self.ee_idx
        if joint_angles is None:
            joint_angles = self.get_joint_angles()

        assert len(self.arm_idxs) == len(joint_angles)
        joint_angles = np.asarray([*joint_angles, 0, 0])

        return super().get_jacobian(link_idx, joint_angles, local_position)[:, :-2]


if __name__ == "__main__":
    world = BulletWorld(gui=True)
    robot: Panda = world.load_robot("robot", robot_class=Panda)
    robot.forward_kinematics([0, 0, 0, 0, 0, 0, 0])
    robot.set_joint_angles([0, 0, 0, 0, 0, 0, 0])
    input()
