from ..base import *
from ..utils.utils import PANDA_URDF

class Panda(Robot):
    urdf_path = PANDA_URDF.as_posix()

    def __init__(self, physics_client: BulletClient, body_uid: int, name: str):
        self.arm_idxs = range(7)
        self.finger_idxs = [8, 9]
        super().__init__(
            physics_client=physics_client,
            body_uid=body_uid,
            name=name,
            ee_idx=10
        )
        self.max_opening_width = 0.08
        self.arm_lower_limit = self.joint_lower_limit[self.arm_idxs]
        self.arm_upper_limit = self.joint_upper_limit[self.arm_idxs]
        self.arm_central = (self.arm_lower_limit + self.arm_upper_limit)/2
        self.open()
        self.ctrl_mode = "pos" # ["pos", "vel", "torque"] 

    def get_joint_angles(self):
        return super().get_joint_angles()[self.arm_idxs]
    
    def get_joint_velocities(self):
        return super().get_joint_velocities()[self.arm_idxs]

    def get_random_arm_angles(self):
        q = np.random.uniform(low=self.arm_lower_limit,
                              high=self.arm_upper_limit)
        return q

    def set_ctrl_mode(self, mode):
        if mode == "torque":
            self.physics_client.setJointMotorControlArray(
                self.uid, 
                jointIndices=list(self.arm_idxs), 
                controlMode=self.physics_client.VELOCITY_CONTROL, 
                forces=np.zeros(len(self.arm_idxs))
            )

    def set_joint_angles(self, angles: np.ndarray):
        for i, angle in zip(self.arm_idxs, angles):
            super().set_joint_angle(joint=i, angle=angle)
    
    def set_joint_torques(self, torques: np.ndarray, gravity_comp=True):
        torques = np.hstack([torques,])
        if gravity_comp == True:
            states = self.physics_client.getJointStates(
                self.uid,
                jointIndices=list(range(self.n_joints)),
            )
            all_joint_types = [self.info[i]["joint_type"] for i in list(self.info)]
            free_joints = [i for i, j in enumerate(all_joint_types) if j != 4]
            q, qdot, qddot = [], [], []
            for j in free_joints:
                q.append(states[j][0])
                qdot.append(states[j][1])
                qddot.append(0.)

            comp_torque = self.physics_client.calculateInverseDynamics(
                self.uid, 
                q, qdot, qddot
            )[:-2]
            torques = torques + comp_torque

        self.physics_client.setJointMotorControlArray(
            self.uid, 
            jointIndices=list(self.arm_idxs), 
            controlMode=self.physics_client.TORQUE_CONTROL, 
            forces=torques
        )

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
        result = super().inverse_kinematics(
            pos, pose, tol, max_iter, start_central
        )
        if result is not None:
            return result[self.arm_idxs]
        return None

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
