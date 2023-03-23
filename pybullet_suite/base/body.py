import numpy as np
import trimesh
from contextlib import contextmanager
from typing import Tuple, Optional
from pybullet_utils.bullet_client import BulletClient
from .tf import Rotation, Pose

# TODO: use dataclass or namedtuple
JOINT_ATRTIBUTE_NAMES = \
    ("joint_index", "joint_name", "joint_type",
     "q_index", "u_index", "flags",
     "joint_damping", "joint_friction", "joint_lower_limit",
     "joint_upper_limit", "joint_max_force", "joint_max_velocity",
     "link_name", "joint_axis", "parent_frame_pos", "parent_frame_orn", "parent_index")


class Body:
    def __init__(self, physics_client: BulletClient, body_uid: int, name:str):
        self.physics_client = physics_client
        self.uid = body_uid
        self.name = name
        self.info = {}
        self.n_joints = self.physics_client.getNumJoints(self.uid)
        for i in range(self.n_joints):
            joint_info = self.physics_client.getJointInfo(self.uid, i)
            self.info[i] = {name: value for name, value in zip(
                JOINT_ATRTIBUTE_NAMES, joint_info)}

    @classmethod
    def from_urdf(
        cls,
        physics_client: BulletClient,
        name:str, 
        urdf_path: str,
        pose: Pose = Pose.identity(),
        use_fixed_base: bool = False,
        scale: float = 1.0
    ) -> "Body":
        body_uid = physics_client.loadURDF(
            fileName=str(urdf_path),
            basePosition=pose.trans,
            baseOrientation=pose.rot.as_quat(),
            useFixedBase=use_fixed_base,
            globalScaling=scale,
        )
        return cls(physics_client, body_uid, name=name)

    @classmethod
    def from_mesh(
        cls,
        physics_client:BulletClient,
        name:str,
        col_path: str,
        viz_path: str,
        pose: Pose = Pose.identity(),
        offset_pose: Pose = Pose.identity(),
        mass: Optional[float] = None,
        scale: float = 1.0,
        rgba_color: Optional[list] = None,
    ) -> "Body":
        # mesh = trimesh.load(viz_path)
        # center = -mesh.bounding_box.centroid
        #tf = Pose.from_matrix(mesh.principal_inertia_transform.copy())
        # com_pos = mesh.center_mass
        # com_orn = tf.rot
        viz_id = physics_client.createVisualShape(
            physics_client.GEOM_MESH,
            fileName=viz_path,
            meshScale=np.ones(3)*scale,
            visualFramePosition=offset_pose.trans,
            # visualFrameOrientation=offset_pose.rot.as_quat(),
            rgbaColor=rgba_color
        )
        # com_orn.as_euler()
        col_id = physics_client.createCollisionShape(
            physics_client.GEOM_MESH,
            fileName=col_path,
            meshScale=np.ones(3)*scale,
            collisionFramePosition=offset_pose.trans,
            # collisionFrameOrientation=offset_pose.rot.as_quat()
        )
        body_uid = physics_client.createMultiBody(
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=viz_id,
            basePosition=pose.trans,
            baseOrientation=pose.rot.as_quat(),
            baseMass=mass,
            # baseInertialFramePosition=-center,
            # baseInertialFrameOrientation=offset_pose.rot.as_quat()
        )
        return cls(physics_client, body_uid, name=name)

    @contextmanager
    def no_set_pose(self, no_viz=False):
        """Context manager not to set the pose of the body
        usage:
        with body.no_set_pose():
            #do something

        Args:
            no_viz (bool, optional): Set True if visualization within the code is not needed. Defaults to False.
        """
        if no_viz:
            self.physics_client.configureDebugVisualizer(
                self.physics_client.COV_ENABLE_RENDERING,
                0
            )
        pose_ = self.get_base_pose()
        yield
        self.set_base_pose(pose_)
        if no_viz:
            self.physics_client.configureDebugVisualizer(
                self.physics_client.COV_ENABLE_RENDERING,
                1
            )

    def get_base_pose(self) -> Pose:
        """Get the base pose of the body

        Returns:
            Pose: The pose of the body base
        """
        pos, orn = self.physics_client.getBasePositionAndOrientation(self.uid)
        return Pose(Rotation.from_quat(orn), np.asarray(pos))

    def set_base_pose(self, pose: Pose):
        """Set base pose of the body

        Args:
            pose (Pose): Pose to set
        """
        self.physics_client.resetBasePositionAndOrientation(
            self.uid, pose.trans, pose.rot.as_quat()
        )

    def get_base_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current velocity of the body

        Returns:
            Tuple[np.ndarray, np.ndarray]: linear, angular velocity
        """
        linear, angular = self.physics_client.getBaseVelocity(self.uid)
        return linear, angular

    def get_link_pose(self, link: int) -> Pose:
        """Get the pose of a specific link.

        Args:
            link (int): target link index

        Returns:
            Pose: the pose of the link
        """
        assert self.n_joints > 0, "This body has no link index: use get_base_pose instead"
        pos, orn = self.physics_client.getLinkState(self.uid, link)[:2]
        return Pose(
            rot=Rotation.from_quat(orn), trans=pos
        )

    def get_AABB(self, output_center_extent=False) -> Tuple[np.ndarray, np.ndarray]:
        """Get Axis-aligned bounding box(AABB) of the body

        Args:
            output_center_extent (bool, optional): True if representation is center/extent. Else, lower/upper. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: lower-upper, or center-extent
        """
        link_num = len(self.info)
        if link_num == 0:
            lower, upper = self.physics_client.getAABB(self.uid, linkIndex=-1)
            lower, upper = np.array(lower), np.array(upper)
        else:
            lowers, uppers = [], []
            for link in [-1, *range(link_num)]:
                lower, upper = \
                    self.physics_client.getAABB(self.uid, linkIndex=link)
                lowers.append(lower)
                uppers.append(upper)
            lower = np.min(lowers, axis=0)
            upper = np.max(uppers, axis=0)
        if output_center_extent:
            center = np.mean(lower+upper)
            extent = upper - lower
            return center, extent
        return lower, upper

    def get_AABB_wrt_obj_frame(self, output_center_extent=True):
        # only for base link
        p = self.physics_client
        collision_data = p.getCollisionShapeData(
            self.uid, -1
        )[0]
        # geometry_type
        if collision_data[2] == p.GEOM_BOX:
            extents = np.array(collision_data[3])
        center = np.array([0, 0, 0])
        if output_center_extent:
            return center, extents
        else:
            half_extents = extents/2
            lower = center - half_extents
            upper = center + half_extents
            return lower, upper

    def get_link_velocity(self, link: int):
        raise NotImplementedError("not implemented")

    def get_joint_angle(self, joint: int) -> float:
        """Get a specific joint angle of the body

        Args:
            joint (int): a joint index

        Returns:
            float: a joint angle.
        """
        assert self.n_joints > 0, "This body has no joint index"
        return self.physics_client.getJointState(self.uid, joint)[0]

    def get_joint_angles(self) -> np.ndarray:
        """Get all joint angles of the body

        Returns:
            np.ndarray: all joint angles of the body
        """
        assert self.n_joints > 0, "This body has no joint index"
        joint_angles = []
        for i in range(self.n_joints):
            joint_angles.append(self.get_joint_angle(i))
        return np.asarray(joint_angles)

    def get_joint_velocity(self, joint: int) -> float:
        """Get a specific joint velocity of the body

        Args:
            joint (int): a joint index

        Returns:
            float: the joint velocity of the spesific joint
        """
        assert self.n_joints > 0, "This body has no joint index"
        return np.asarray(self.physics_client.getJointState(self.uid, joint)[1])

    def get_joint_velocities(self) -> np.ndarray:
        """Get all joint angles of the body

        Returns:
            np.ndarray: all joint angles of the body
        """
        assert self.n_joints > 0, "This body has no joint index"
        joint_velocities = []
        for i in range(self.n_joints):
            joint_velocities.append(self.get_joint_velocity(i))
        return np.asarray(joint_velocities)

    def set_joint_angle(self, joint: int, angle: float):
        """Set a joint angle of the body

        Args:
            joint (int): a joint index
            angle (float): the angle of the joint
        """
        assert self.n_joints > 0, "This body has no joint index"
        self.physics_client.resetJointState(
            self.uid, jointIndex=joint, targetValue=angle)

    def set_joint_angles(self, angles: np.ndarray):
        """Set all joint angles of the body

        Args:
            angles (np.ndarray): all joint angles
        """
        assert self.n_joints > 0, "This body has no joint index"
        assert len(angles) == self.n_joints
        for i, angle in zip(range(self.n_joints), angles):
            self.set_joint_angle(joint=i, angle=angle)

    def __repr__(self):
        return f"Body:{self.name}"


if __name__ == "__main__":
    pass
