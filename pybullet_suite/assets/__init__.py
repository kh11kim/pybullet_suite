import os

dir_path = os.path.dirname(os.path.realpath(__file__))

DICE_URDF = dir_path + "/urdfs/dice/urdf/dice.urdf"
DICE_GRASP_SET = dir_path + "/urdfs/dice/grasp_set.npz"

PANDA_URDF = dir_path + "/urdfs/panda/franka_panda.urdf"
HAND_URDF = dir_path + "/urdfs/panda/hand.urdf"