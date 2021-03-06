import time
from pybullet_suite.base import *
from pybullet_suite.utils.scene_maker import BulletSceneMaker
from pybullet_suite.robots.panda import Panda
from itertools import combinations

world = BulletWorld(gui=True)
robot: Panda = world.load_robot("robot", robot_class=Panda)

robot.set_arm_angles([0,0,0,-1,-1,-1,0])


sm = BulletSceneMaker(world)
box = sm.create_box("table", [0.5, 0.5, 0.5], 1,
position=[0,0,0])
# get_contact_points
tic = time.time()
#world.step(only_collision_detection=True)

points = world.physics_client.getContactPoints(robot.uid)
elapsed = time.time() - tic
print(f"elapsed time for collision check: {elapsed}") # 0.005984


# get_closest_points
tic = time.time()
result = world.is_self_collision(robot)
result2 = world.is_body_pairwise_collision(robot, box)

#results = [CollisionInfo(*point) for point in points]
elapsed = time.time() - tic
print(f"elapsed time for collision check: {elapsed}") # 0.001995

input()