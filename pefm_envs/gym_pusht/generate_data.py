"""
Download lerobot/pusht_keypoints from HuggingFace and convert to PEFM NPZ format.

Output per timestep:
  eef_pos: (18,) = keypoints(16D) + agent_pos(2D)
  action:  (2,)  = target position [x, y]
  pc:      (1,3) = dummy zeros (pipeline compat)

Usage:
  python -m pefm_envs.gym_pusht.generate_data \
      --data_out_dir ../data/pusht_gym --num_demos 200
"""

import os
import argparse
import numpy as np
from tqdm import tqdm


def download_and_convert(data_out_dir, num_demos=None, dataset_name="lerobot/pusht_keypoints"):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("pip install datasets")

    print(f"Loading {dataset_name} from HuggingFace...")
    ds = load_dataset(dataset_name, split="train")
    print(f"Dataset loaded: {len(ds)} frames")

    # Check available columns
    columns = ds.column_names
    print(f"Columns: {columns}")

    has_env_state = "observation.environment_state" in columns
    has_state = "observation.state" in columns
    has_action = "action" in columns
    has_episode = "episode_index" in columns

    if not has_action or not has_episode:
        raise ValueError(f"Dataset missing required columns. Found: {columns}")

    if not has_env_state and not has_state:
        raise ValueError("Need observation.environment_state or observation.state")

    # Group by episode
    episode_indices = np.array(ds["episode_index"])
    unique_episodes = np.unique(episode_indices)
    print(f"Found {len(unique_episodes)} episodes")

    if num_demos is not None and num_demos < len(unique_episodes):
        unique_episodes = unique_episodes[:num_demos]
        print(f"Using first {num_demos} episodes")

    pcs_dir = os.path.join(data_out_dir, "pcs")
    os.makedirs(pcs_dir, exist_ok=True)

    total_frames = 0
    for ep_idx in tqdm(unique_episodes, desc="Converting episodes"):
        # Get frames for this episode
        mask = episode_indices == ep_idx
        frame_indices = np.where(mask)[0]

        for t, global_idx in enumerate(frame_indices):
            row = ds[int(global_idx)]

            # Build 18D state: keypoints(16) + agent_pos(2)
            if has_env_state:
                env_state = np.array(row["observation.environment_state"], dtype=np.float32)
                # Should be 16D keypoints
                if len(env_state) != 16:
                    raise ValueError(f"Expected 16D environment_state, got {len(env_state)}")
            else:
                # Fallback: pad state to 16D with zeros
                env_state = np.zeros(16, dtype=np.float32)

            if has_state:
                agent_pos = np.array(row["observation.state"], dtype=np.float32)[:2]
            else:
                agent_pos = np.zeros(2, dtype=np.float32)

            eef_pos = np.concatenate([env_state, agent_pos])  # (18,)

            action = np.array(row["action"], dtype=np.float32)[:2]  # (2,)
            pc = np.zeros((1, 3), dtype=np.float32)  # dummy

            fn = os.path.join(pcs_dir, f"pusht_ep{ep_idx:06d}_view0_t{t:04d}.npz")
            np.savez(fn, eef_pos=eef_pos, action=action, pc=pc)
            total_frames += 1

    print(f"Saved {total_frames} frames from {len(unique_episodes)} episodes to {pcs_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_out_dir", type=str, required=True)
    parser.add_argument("--num_demos", type=int, default=None,
                        help="Max episodes to convert (default: all)")
    parser.add_argument("--dataset", type=str, default="lerobot/pusht_keypoints",
                        help="HuggingFace dataset name")
    args = parser.parse_args()

    download_and_convert(args.data_out_dir, args.num_demos, args.dataset)


if __name__ == "__main__":
    main()
