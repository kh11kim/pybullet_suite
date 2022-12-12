from ..base import *
from ..utils.scene_maker import *
from ..utils.utils import HAND_URDF

class Gripper:
    def __init__(self, world: BulletWorld, sm: BulletSceneMaker, finger_swept_vol=False):
        self.world = world
        self.sm = sm
        self.body = None
        self.urdf_path = HAND_URDF
        self.max_opening_width = 0.08
        self.T_body_tcp = Pose(Rotation.identity(), [0.0, 0.0, 0.103])
        self.T_tcp_body = self.T_body_tcp.inverse()
        self.finger_swept_vol = finger_swept_vol

        self.remove_pose = Pose(trans=[10,10,10]) * self.T_tcp_body
        self.w = 0.0085
        # self.T_tcp_finger_col_box1 = Pose(trans=[0, 0.04+self.w, 0])
        # self.T_tcp_finger_col_box2 = Pose(trans=[0, -0.04-self.w, 0])

    def reset(self, T_world_tcp: Pose, name=None):
        if name is None:
            self.name = "hand"
        else:
            self.name = name
        T_world_body = T_world_tcp * self.T_tcp_body
        if not self.name in self.world.bodies:
            self.body = self.world.load_urdf(self.name, self.urdf_path, T_world_body)
            if self.finger_swept_vol:
                self.grasping_box = self.sm.create_box("grasping_box", [0.0085, 0.04, 0.0085], 0.01, pose=T_world_tcp, rgba_color=[0,1,0,0.4])
            #self.col_box1 = self.sm.create_box("col_box1", [self.w]*3, 0.01, pose=T_world_tcp*self.T_tcp_finger_col_box1, rgba_color=[1,0,0,0.4])
            #self.col_box2 = self.sm.create_box("col_box2", [self.w]*3, 0.01, pose=T_world_tcp*self.T_tcp_finger_col_box2, rgba_color=[1,0,0,0.4])
        else:
            self.body.set_base_pose(T_world_body)
            #self.col_box1.set_base_pose(T_world_tcp*self.T_tcp_finger_col_box1)
            #self.col_box2.set_base_pose(T_world_tcp*self.T_tcp_finger_col_box2)
            if self.finger_swept_vol:
                self.grasping_box.set_base_pose(T_world_tcp)
        self.grip()
    
    def is_grasp_candidate(self, obj:Body):
        is_in_swept_vol = self.world.is_body_pairwise_collision(self.grasping_box, obj)
        is_col_gripper = self.world.is_body_pairwise_collision(self.body, obj)
        # is_col_finger1 = self.world.is_body_pairwise_collision(self.col_box1, obj)
        # is_col_finger2 = self.world.is_body_pairwise_collision(self.col_box2, obj)
        return is_in_swept_vol and not is_col_gripper

    def get_tcp_pose(self):
        T_world_body = self.body.get_base_pose()
        return T_world_body * self.T_body_tcp

    def remove(self):
        self.body.set_base_pose(self.remove_pose)
        #self.world.remove_body(name=self.name)

    def detect_contact(self):
        self.world.step(only_collision_detection=True)
        if self.world.get_contacts(body=self.body):
            return True
        return False
    
    def detect_collision(self, obstacles: List[str]):
        for obs_name in obstacles:
            if self.world.is_body_pairwise_collision(self.name, obstacles):
                return True
        return False

    # def set_ctrl_mode(self, mode):
    #     if mode == "torque":
    #         self.physics_client.setJointMotorControlArray(
    #             self.uid, 
    #             jointIndices=list(self.arm_idxs), 
    #             controlMode=self.physics_client.VELOCITY_CONTROL, 
    #             forces=np.zeros(len(self.arm_idxs))
    #         )

    def grip(self, width=None, control=False, force=5):
        assert self.body is not None
        if width is None:
            width = self.max_opening_width
        if not control:
            self.body.set_joint_angle(0, width / 2)
            self.body.set_joint_angle(1, width / 2)
        else:
            #self.col_box1.set_base_pose(self.remove_pose*Pose(trans=[1,0,0]))
            #self.col_box2.set_base_pose(self.remove_pose*Pose(trans=[1.5,0,0]))
            if self.finger_swept_vol:
                self.grasping_box.set_base_pose(self.remove_pose*Pose(trans=[2,0,0]))
            self.world.physics_client.setJointMotorControlArray(
                self.body.uid, 
                jointIndices=[0, 1], 
                controlMode=self.world.physics_client.POSITION_CONTROL, 
                forces=[force, force]
            )

    def close(self):
        assert self.body is not None
        self.body.set_joint_angle(0, 0)
        self.body.set_joint_angle(1, 0)

if __name__ == "__main__":
    world = BulletWorld(gui=True)
    hand = Gripper(world)
    T_tcp = Pose(Rotation.identity(), [0,0,0.5])
    hand.reset(T_tcp)
    hand.grip()
    hand.grip(0)
    