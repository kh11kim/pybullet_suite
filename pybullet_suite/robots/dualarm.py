from ..base import *
from .panda import Panda

class DualArm:
    def __init__(self, robot1: Panda, robot2: Panda):
        self.robot1 = robot1
        self.robot2 = robot2
        self.arm_lower_limit = np.hstack([self.robot1.arm_lower_limit, self.robot1.arm_lower_limit])
        self.arm_upper_limit = np.hstack([self.robot1.arm_upper_limit, self.robot1.arm_upper_limit])
        self.arm_central = np.hstack([self.robot1.arm_central, self.robot1.arm_central])

    
    def get_joint_angles(self):
        q1 = self.robot1.get_joint_angles()
        q2 = self.robot2.get_joint_angles()
        q = np.hstack([q1, q2])
        return q
    
    def get_ee_pose(self):
        T1 = self.robot1.get_ee_pose()
        T2 = self.robot2.get_ee_pose()
        return [T1, T2]
    
    @property
    def arm_central(self):
        q = self.robot1.arm_central
        return np.hstack([q, q])