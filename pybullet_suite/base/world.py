from contextlib import contextmanager
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Union
import pybullet as p
from itertools import combinations
from pybullet_utils.bullet_client import BulletClient
from .body import Body
from .camera import Camera, CameraIntrinsic
from .tf import Rotation, Pose
from .robot import Robot
import io

SOLVER_ITERATIONS = 150
COLLISION_DISTANCE = 0.

@dataclass
class Contact:
    bodyA: Body
    bodyB: Body
    point: np.ndarray
    normal: np.ndarray
    depth: float
    force: float

@dataclass
class DistanceInfo:
    contact_flag: bool
    bodyA: int
    bodyB: int
    linkA: int
    linkB: int
    position_on_A: List[float]
    position_on_B: List[float]
    contact_normal_on_B: List[float]
    contact_distnace: float
    normal_force: float
    lateral_frictionA: float
    lateral_friction_dirA: List[float]
    lateral_frictionB: float
    lateral_friction_dirB: List[float]


class BulletWorld:
    def __init__(
        self,
        gui: bool = False,
        dt = 1.0 / 1000.0,
        dt_gui = 1.0/100.,
        background_color = None
    ):
        self.gui = gui
        self.dt = dt
        self.dt_gui = dt_gui
        self._bodies = {}
        self._frames = {}
        self._body_names = {}
        self.const_id = {}
        self.temp_stdout = io.BytesIO()

        connection_mode = p.GUI if gui else p.DIRECT

        options = ""
        if background_color is not None:
            background_color = np.array(background_color).astype(np.float64) / 255
            options = f"--background_color_red={background_color[0]} \
                        --background_color_green={background_color[1]} \
                        --background_color_blue={background_color[2]}"
            
            self.physics_client = BulletClient(
                    connection_mode=connection_mode,
                    options=options)
        else:
            self.physics_client = BulletClient(
                    connection_mode=connection_mode)
        
        self.physics_client.setTimeStep(self.dt)
        self.set_debug_visualizer(False)
        self.reset()
        #self.set_gravity([0,0,-9.8])

    def set_debug_visualizer(self, onoff:bool=False):
        # or type "g" in debug window
        p = self.physics_client
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, onoff)
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, onoff)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, onoff)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, onoff)

    def set_view(
        self,
        eye_point:np.ndarray, 
        target_point:Optional[np.ndarray]=np.zeros(3)
    ):
        delta = np.array(eye_point) - np.array(target_point)
        d = np.linalg.norm(delta)
        yaw = (np.arctan2(delta[1], delta[0]) +np.pi/2) *180/np.pi
        pitch = -np.arctan2(delta[2], delta[0]) *180/np.pi
        p.resetDebugVisualizerCamera(
            d, yaw, pitch, target_point)
    
    @property
    def bodies(self)->Dict[str, Body]:
        return self._bodies

    @property
    def frames(self)->Dict[str, Body]:
        return self._frames

    @property
    def body_names(self)->Dict[int, str]:
        return self._body_names
    
    @contextmanager
    def no_rendering(self):
        self.physics_client.configureDebugVisualizer(
            self.physics_client.COV_ENABLE_RENDERING, 
            0
        )
        yield
        self.physics_client.configureDebugVisualizer(
            self.physics_client.COV_ENABLE_RENDERING, 
            1
        )

    ## ------------------------------------------------------------
    ## Simulation
    ## ------------------------------------------------------------
    def reset(self):
        self.physics_client.resetSimulation()
        self.physics_client.setPhysicsEngineParameter(
            fixedTimeStep=self.dt,
            numSolverIterations=SOLVER_ITERATIONS
        )
        self._bodies = {}
        self._body_names = {}
        self.sim_time = 0.0
        self.last_render_time = 0.0

    def step(self, only_collision_detection=False):
        if only_collision_detection:
            self.physics_client.performCollisionDetection()
        else:
            self.physics_client.stepSimulation()
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
        robot_class,
        name: str,
        pose: Pose = Pose.identity(),
        
    ):
        assert name not in self.bodies
        assert Robot in robot_class.__bases__
        robot = robot_class.make(
            self.physics_client, 
            pose=pose,
            name=name
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
            name=name,
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
    
    def load_mesh(
        self,
        name: str,
        viz_path: str,
        col_path: str,
        pose: Pose,
        mass=0.01,
        mesh_scale=1,
        rgba_color=None,
    ):
        body = Body.from_mesh(
            physics_client=self.physics_client, 
            name=name,
            col_path=col_path, 
            viz_path=viz_path,
            pose=pose,
            mass=mass,
            scale=mesh_scale,
            rgba_color=rgba_color
        )
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
    
    #---------------collision detection----------------------
    def get_closest_points(
        self, 
        body1: Body, body2: Body, 
        link1:int=None, link2:int=None,
        d:float=None
    ):
        if d is None:
            d = COLLISION_DISTANCE
        if (link1 is not None) & (link2 is not None):
            results = self.physics_client.getClosestPoints(
                bodyA=body1.uid, bodyB=body2.uid, 
                linkIndexA=link1, linkIndexB=link2,
                distance=d)
        elif (link1 is None) & (link2 is None):
            results = self.physics_client.getClosestPoints(
                bodyA=body1.uid, bodyB=body2.uid, 
                distance=d)
        return [DistanceInfo(*info) for info in results]
    
    def get_body_pairwise_distance(
        self, body: Union[str, Body], obstacles: Union[List[str], List[Body]], all=False, **kwargs)->bool:
        """Get boolean whether two body is in collision

        Args:
            body (Union[str, Body]): target body
            obstacles (Union[List[str], List[Body]]): a list of obstacles. One body is also available.

        Returns:
            bool: True if the body has a collision with obstacles
        """
        if isinstance(body, str):
            body = self.bodies[body]

        if all:
            obstacles = [self.bodies[name] for name in self.bodies]
        else:
            if not isinstance(obstacles, List):
                obstacles = [obstacles]
            if isinstance(obstacles[0], str):
                obstacles = [self.bodies[name] for name in obstacles]
        collisions = []
        for other in obstacles:
            if body.uid != other.uid:
                collisions += [*self.get_closest_points(body, other, **kwargs)]
        return collisions

    def is_body_pairwise_collision(
        self, body: Union[str, Body], obstacles: Union[List[str], List[Body]], all=False, **kwargs)->bool:
        """Get boolean whether two body is in collision

        Args:
            body (Union[str, Body]): target body
            obstacles (Union[List[str], List[Body]]): a list of obstacles. One body is also available.

        Returns:
            bool: True if the body has a collision with obstacles
        """
        if isinstance(body, str):
            body = self.bodies[body]

        if all:
            obstacles = [self.bodies[name] for name in self.bodies]
        else:
            if not isinstance(obstacles, List):
                obstacles = [obstacles]
            if isinstance(obstacles[0], str):
                obstacles = [self.bodies[name] for name in obstacles]
        

        return any(self.get_closest_points(body, other, **kwargs) \
            for other in obstacles if body.uid != other.uid)

    def is_link_pairwise_collision(
        self, 
        body1: Union[str, Body], 
        body2: Union[str, Body],
        link1: int,
        link2: int
    ):
        if isinstance(body1, str):
            body1 = self.bodies[str]
        if isinstance(body1, str):
            body2 = self.bodies[str]
        point = self.get_closest_points(
            body1=body1, body2=body2, 
            link1=link1, link2=link2)
        if point:
            return True
        return False
    
    def get_link_pairwise_collision(
        self, 
        body1: Union[str, Body], 
        body2: Union[str, Body],
        link1: int,
        link2: int,
        d: float=0.0
    ):
        if isinstance(body1, str):
            body1 = self.bodies[str]
        if isinstance(body1, str):
            body2 = self.bodies[str]
        point = self.get_closest_points(
            body1=body1, body2=body2, 
            link1=link1, link2=link2, d=d)
        if point:
            return point[0]
        return None

    def is_self_collision(self, robot: Union[str, Robot])-> bool:
        """Get boolean whether the robot is in self-collision

        Args:
            robot (Union[str, Robot]): The robot

        Returns:
            bool: True if the robot is in self-collision.
        """
        if isinstance(robot, str):
            robot = self.bodies[robot]

        for link1, link2 in combinations(robot.info.keys(), 2):
            adjacent = (link1 == robot.info[link2]["parent_index"])
            if not adjacent:
                is_col = self.is_link_pairwise_collision(
                    robot, robot, link1, link2)
                if is_col:
                    return True
        return False
                
    def get_body_pairwise_contacts(
        self,
        body1: Union[str, Body],
        body2: Union[str, Body],
    ):
        if isinstance(body1, str):
            body1 = self.bodies[body1]
        if isinstance(body2, str):
            body2 = self.bodies[body2]
        
        points = self.physics_client.getContactPoints(body1.uid, body2.uid)
        return points

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

    # def draw_workspace(self, size, mid):
    #     points = workspace_lines(size, mid)
    #     color = [0.5, 0.5, 0.5]
    #     for i in range(0, len(points), 2):
    #         self.physics_client.addUserDebugLine(
    #             lineFromXYZ=points[i], lineToXYZ=points[i + 1], lineColorRGB=color
    #         )
    
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
    
    def make_fixed_constraint(self, body1: str, body2: str):
        body1_uid = self.bodies[body1]
        body2_uid = self.bodies[body2]
        const_id = self.physics_client.createConstraint(
            parentBodyUniqueId=body1_uid,
            parentLinkIndex=-1,
            childBodyUniqueId=body2_uid,
            childLinkIndex=-1,
            jointType=self.physics_client.JOINT_FIXED,
            jointAxis=[1,0,0]
        )
        const_str = f"{body1}_{body2}_fixed"
        self.const_id[const_str] = const_id
    
    def watch_workspace(self, target_pos, distance = 1.0, cam_yaw=20, cam_pitch=-10):
        self.physics_client.resetDebugVisualizerCamera(
            cameraDistance=distance,
            cameraYaw=cam_yaw,
            cameraPitch=cam_pitch,
            cameraTargetPosition=target_pos
        )

if __name__ == "__main__":
    world = BulletWorld(gui=True)
    panda = world.load_urdf("panda", "data/urdfs/panda/franka_panda.urdf")
    world.set_view([0.5, 0., 0.5], [0,0,0])
    box_pose = Pose(Rotation.identity(), [0.3, 0, 0.1])
    box = world.load_urdf("box", "data/urdfs/blocks/cube.urdf", box_pose, scale=5.)
    input()