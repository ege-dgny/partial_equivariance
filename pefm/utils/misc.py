import numpy as np
import torch


def to_torch(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def rotate_around_z(
    points,
    angle_rad=0.0,
    center=np.array([0.0, 0.0, 0.0]),
    scale=np.array([1.0, 1.0, 1.0]),
):
    assert (len(points.shape) == 1 and len(points) == 3) or points.shape[-1] == 3
    p_shape = points.shape
    points = points.reshape(-1, 3) - center[None]

    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)
    rotation_matrix = np.array(
        [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]]
    )

    rotated_points = np.dot(points, rotation_matrix.T) * scale[None] + center[None]
    rotated_points = rotated_points.reshape(p_shape)

    return rotated_points


def get_env_class(env_name):
    if env_name == "fold":
        from equibot.envs.sim_mobile.folding_env import FoldingEnv
        return FoldingEnv
    elif env_name == "cover":
        from equibot.envs.sim_mobile.covering_env import CoveringEnv
        return CoveringEnv
    elif env_name == "close":
        from equibot.envs.sim_mobile.closing_env import ClosingEnv
        return ClosingEnv
    elif env_name == "pour":
        from pefm_envs.sim_mobile.pouring_env import PouringEnv
        return PouringEnv
    elif env_name == "insert":
        from pefm_envs.sim_mobile.insertion_env import InsertionEnv
        return InsertionEnv
    elif env_name == "compass_close":
        from pefm_envs.sim_mobile.compass_closing_env import CompassClosingEnv
        return CompassClosingEnv
    # Franka single-arm tasks
    elif env_name == "pick_place":
        from pefm_envs.sim_franka.pick_place_env import PickPlaceEnv
        return PickPlaceEnv
    elif env_name == "peg_insert":
        from pefm_envs.sim_franka.peg_insert_env import PegInsertEnv
        return PegInsertEnv
    elif env_name == "centering":
        from pefm_envs.sim_franka.centering_env import CenteringEnv
        return CenteringEnv
    elif env_name == "book_insert":
        from pefm_envs.sim_franka.book_insert_env import BookInsertEnv
        return BookInsertEnv
    elif env_name == "cup_pour":
        from pefm_envs.sim_franka.cup_pour_env import CupPourEnv
        return CupPourEnv
    elif env_name == "push_t":
        from pefm_envs.sim_franka.push_t_env import PushTEnv
        return PushTEnv
    elif env_name == "push_t_gym":
        from pefm_envs.gym_pusht.pusht_wrapper import GymPushTEnv
        return GymPushTEnv
    else:
        raise ValueError(f"Environment [{env_name}] not found.")


def get_dataset(cfg, mode="train"):
    from pefm.datasets.dataset import BaseDataset
    return BaseDataset(cfg.data.dataset, mode)


def get_agent(agent_name):
    if agent_name == "pefm":
        from pefm.agents.pefm_agent import PEFMAgent
        return PEFMAgent
    elif agent_name == "dp":
        from equibot.policies.agents.dp_agent import DPAgent
        return DPAgent
    else:
        raise ValueError(f"Agent with name [{agent_name}] not found.")
