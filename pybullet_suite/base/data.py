import pkg_resources

dirpath = pkg_resources.resource_filename("pybullet_suite", 'assets')

PANDA_URDF = dirpath + "/urdfs/panda/franka_panda.urdf"
HAND_URDF = dirpath + "/urdfs/panda/hand.urdf"
