"""
Wrapper around gym-pusht for PEFM pipeline compatibility.

obs_type="environment_state_agent_pos" gives:
  - keypoints: 16D (8 T-block keypoints as [x0,y0,...,x7,y7])
  - agent_pos: 2D [x, y]
Concatenated as 18D eef_pos. Action is 2D target position.
Dummy pc=(1,3) for pipeline compatibility.
"""

import numpy as np
import gymnasium as gym

try:
    import gym_pusht  # noqa: F401 — registers the env
except ImportError:
    raise ImportError("pip install gym-pusht")


class GymPushTEnv:
    """PEFM-compatible wrapper for gym-pusht/PushT-v0."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.max_episode_length = cfg.max_episode_length
        self.num_eef = getattr(cfg, "num_eef", 1)
        self.dof = getattr(cfg, "dof", 2)

        self.env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="environment_state_agent_pos",
            render_mode="rgb_array",
        )
        self._step_count = 0
        self._last_reward = 0.0

    def _obs_to_state(self, obs):
        """Convert gym obs dict → 18D state vector."""
        kp = np.array(obs["environment_state"], dtype=np.float32)  # (16,)
        agent = np.array(obs["agent_pos"], dtype=np.float32)       # (2,)
        return np.concatenate([kp, agent])  # (18,)

    def reset(self):
        obs, info = self.env.reset()
        self._step_count = 0
        self._last_reward = 0.0
        return self._obs_to_state(obs)

    def step(self, action, dummy_reward=False):
        """
        action: (2,) target [x, y] position.
        Returns: state, reward, done, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action[:2])
        self._step_count += 1
        self._last_reward = reward
        done = terminated or truncated or self._step_count >= self.max_episode_length
        return self._obs_to_state(obs), reward, done, info

    def render(self):
        frame = self.env.render()
        return {
            "pc": np.zeros((1, 3), dtype=np.float32),
            "images": [frame],
        }

    def compute_reward(self):
        return self._last_reward

    def get(self, attr):
        """For SubprocVecEnv compatibility (e.g. env.get('args'))."""
        if attr == "args":
            return self.cfg
        return getattr(self, attr)

    def close(self):
        self.env.close()
