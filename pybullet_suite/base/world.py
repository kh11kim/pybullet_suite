import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict
import pybullet as p

#from pybullet_suite.base.world import workspace_lines
from pybullet_utils.bullet_client import BulletClient
from pybullet_suite.base.body import Body
from pybullet_suite.base.camera import Camera, CameraIntrinsic
from pybullet_suite.base.tf import Rotation, Pose
from pybullet_suite.base.robot import Robot

SOLVER_ITERATIONS = 150

class BulletWorld:
    def __init__(
        self,
        gui: bool = False,
        dt = 1.0 / 240.0,
        background_color = None
    ):
        self.gui = gui
        self.dt = dt
        self._bodies = {}
        self._body_names = {}
        options = ""
        if background_color is not None:
            background_color = background_color.astype(np.float64) / 255
            options = f"--background_color_red={background_color[0]} \
                        --background_color_green={background_color[1]} \
                        --background_color_blue={background_color[2]}"
        
        connection_mode = p.GUI if gui else p.DIRECT
        self.physics_client = BulletClient(
            connection_mode=connection_mode,
            options=options
        )
        self.reset()
    
    @property
    def bodies(self)->Dict[Body]:
        return self._bodies

    @property
    def body_names(self)->Dict[int]:
        return self._body_names
    
    ## ------------------------------------------------------------
    ## Simulation
    ## ------------------------------------------------------------
    def reset(self):
        self.physics_client.resetSimulation()
        self.physics_client.setPhysicsEngineParameter(
            fixedTimeStep=self.dt,
            numSolverIterations=SOLVER_ITERATIONS
        )
        self.bodies = {}
        self.sim_time = 0.0
    
    def step(self, only_collision_detection=False):
        if only_collision_detection:
            self.physics_client.performCollisionDetection()
        else:
            self.physics_client.stepSimulation()
        if self.gui:
            time.sleep(self.dt)
        self.sim_time += self.dt
    
    def save_state(self):
        return self.physics_client.saveState()
    
    def restore_state(self, state_uid):
        self.physics_client.restoreState(stateId=state_uid)
    
    def close(self):
        self.physics_client.disconnect()
    
    def set_gravity(self, gravity: np.ndarray):
        self.physics_client.setGravity(*gravity)
    
    def load_robot(
        self, 
        name: str,
        pose: Pose = Pose.identity(),
        robot_class = Robot
    ):
        assert name not in self.bodies
        assert Robot in robot_class.__bases__
        robot = robot_class.make(
            self.physics_client, 
            pose=pose
        )
        self.register_body(name, robot)
        return robot

    def load_urdf(
        self, 
        name: str, 
        urdf_path: str, 
        pose: Pose = Pose.identity(),
        use_fixed_base: bool = False,
        scale=1.0
    ) -> Body:
        assert name not in self.bodies, "There are already same body name in the world"

        body = Body.from_urdf(
            physics_client=self.physics_client, 
            urdf_path=urdf_path, 
            pose=pose, 
            use_fixed_base=use_fixed_base,
            scale=scale
        )
        self.register_body(name, body)
        return body

    def load_ycb(
        self,
        name: str,
        path: str, 
        pose: Pose = Pose.identity(),
        scale=1.0
    ) -> Body:
        assert name not in self.bodies, "There are already same body name in the world"
        col_path = path + "/nontextured.stl"
        viz_path = path + "/textured.obj"
        body = Body.from_mesh(
            physics_client=self.physics_client, 
            col_path=col_path, 
            viz_path=viz_path,
            pose=pose,
            mass=0.1,
            scale=scale
        )
        body.set_base_pose(pose)
        self.register_body(name, body)
        return body
    
    def register_body(self, name: str, body: Body):
        self.bodies[name] = body
        self.body_names[body.uid] = name

    def remove_body(
        self, 
        name: Optional[str] = None, 
        body: Optional["Body"] = None
    ):
        assert (name is not None) | (body is not None), "Name or Body should be given."
        if name is not None:
            body = self.bodies[name]
        self.physics_client.removeBody(body.uid)
        self.body_names.pop(body.uid, None)
        self.bodies.pop(name, None)
        
    
    # def add_constraint(self, *argv, **kwargs):
    #     pass
    #     #TODO

    def add_camera(self, intrinsic: CameraIntrinsic, near: float, far: float):
        """Add camera to the world

        Args:
            intrinsic (CameraIntrinsic): Intrinsic parameter class
            near (float): near distance
            far (float): far distance

        Returns:
            Camera: camera class
        """
        camera = Camera(self.physics_client, intrinsic, near, far)
        return camera
    
    def get_contacts(
        self, 
        name: Optional[str] = None, 
        body: Optional["Body"] = None,
        exception: Optional[List["str"]] = None,
        thres: float = 0.
    ):
        assert (name is not None) | (body is not None), "Name or Body should be given."
        self.step(only_collision_detection=True)
        if name is not None:
            body = self.bodies[name]
        points = self.physics_client.getContactPoints(body.uid)
        exception_list = contacts = []
        if exception is not None:
            exception_list = [self.bodies[body_name].uid for body_name in exception]
        for point in points:
            bodyA_name = self.body_names[point[1]]
            bodyB_name = self.body_names[point[2]]
            if point[2] in exception_list:
                continue
            depth = abs(point[8])
            if depth > thres:
                bodyA_name = self.body_names[point[1]]
                bodyB_name = self.body_names[point[2]]
                contact = Contact(
                    bodyA=self.bodies[bodyA_name],
                    bodyB=self.bodies[bodyB_name],
                    point=point[5],
                    normal=point[7],
                    depth=point[8],
                    force=point[9],
                )
                #contacts.append(contact)
                return True
        return False

    def get_stable_z(
        self,
        body: Body,
    ):
        pass

    def draw_workspace(self, size, mid):
        points = workspace_lines(size, mid)
        color = [0.5, 0.5, 0.5]
        for i in range(0, len(points), 2):
            self.physics_client.addUserDebugLine(
                lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
            )
    
    def wait_for_rest(self, timeout=2.0, tol=0.01):
        timeout = self.sim_time + timeout
        is_rest = False
        while not is_rest and self.sim_time < timeout:
            for _ in  range(60):
                self.step()
            is_rest = True
            for _, body in self.bodies.items():
                if np.linalg.norm(body.get_base_velocity()) > tol:
                    is_rest = False
                    break

@dataclass
class Contact(object):
    bodyA: Body
    bodyB: Body
    point: np.ndarray
    normal: np.ndarray
    depth: float
    force: float


if __name__ == "__main__":
    world = BulletWorld(gui=True)
    panda = world.load_urdf("panda", "data/urdfs/panda/franka_panda.urdf")
    box_pose = Pose(Rotation.identity(), [0.3, 0, 0.1])
    box = world.load_urdf("box", "data/urdfs/blocks/cube.urdf", box_pose, scale=5.)
    input()