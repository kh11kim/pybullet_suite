import os
from pathlib import Path

module_path = Path(os.path.realpath(__file__)).parent.parent.parent
asset_path = module_path / "data"
DICE_URDF = asset_path / "urdfs/dice/urdf/dice.urdf"
DICE_GRASP_SET = asset_path / "urdfs/dice/grasp_set.npz"

PANDA_URDF = (asset_path / "urdfs/panda/franka_panda.urdf")
HAND_URDF = (asset_path / "urdfs/panda/hand.urdf")

