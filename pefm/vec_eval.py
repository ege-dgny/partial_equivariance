import os
import sys
import torch
import hydra
import omegaconf
import wandb
import numpy as np
import getpass as gt
from tqdm import tqdm

from pefm.eval import organize_obs
from pefm.utils.media import combine_videos


def run_eval(
    env,
    agent,
    vis=False,
    num_episodes=1,
    log_dir=None,
    reduce_horizon_dim=True,
    verbose=False,
    use_wandb=False,
    ckpt_name=None,
):
    if hasattr(agent, "obs_horizon") and hasattr(agent, "ac_horizon"):
        obs_horizon = agent.obs_horizon
        ac_horizon = agent.ac_horizon
    else:
        obs_horizon = 1
        ac_horizon = 1

    images = []
    obs_history = []
    num_envs = len(env.remotes)
    for i in range(num_envs):
        images.append([])
    state = env.reset()

    env_module_name = env.get_attr("__module__")[0]

    pred_horizon = agent.pred_horizon if hasattr(agent, "pred_horizon") else 1
    rgb_render = render = env.env_method("render")
    obs = organize_obs(render, rgb_render, state)
    for i in range(obs_horizon):
        obs_history.append(obs)
    for i in range(num_envs):
        images[i].append(rgb_render[i]["images"][0][..., :3])
    dof = getattr(agent, "dof", 7)
    if dof > 2:
        grip_state = np.array(obs["state"])[..., -1].reshape(num_envs, -1)[:, -1].astype(
            np.float32
        )

    sample_pc = render[0]["pc"]
    mean_num_points_in_pc = np.mean([len(render[k]["pc"]) for k in range(len(render))])

    done = [False] * num_envs
    if log_dir is not None:
        history = []
        for i in range(num_envs):
            history.append(dict(action=[], eef_pos=[], entropy=[], selector_params=[]))
    t = 0
    pbar = tqdm(
        list(range(env.get("args").max_episode_length // ac_horizon)),
        leave=False,
        desc="Vec Eval",
    )
    while not np.all(done):
        if obs_horizon == 1 and reduce_horizon_dim:
            agent_obs = obs
        else:
            agent_obs = dict()
            for k in obs.keys():
                if k == "pc":
                    agent_obs[k] = [o[k] for o in obs_history[-obs_horizon:]]
                else:
                    agent_obs[k] = np.stack([o[k] for o in obs_history[-obs_horizon:]])

        ac, ac_dict = agent.act(agent_obs, return_dict=True)

        if log_dir is not None:
            for i in range(num_envs):
                history[i]["action"].append(ac[i])
                history[i]["eef_pos"].append(obs["state"][i])
                entry = ac_dict[i] if isinstance(ac_dict, list) else ac_dict
                if entry is not None and "selector_entropy" in entry:
                    history[i]["entropy"].append(float(entry["selector_entropy"]))
                    history[i]["selector_params"].append(np.asarray(entry["selector_params"]))
                else:
                    history[i]["entropy"].append(float("nan"))
                    history[i]["selector_params"].append(None)

        for ac_ix in range(ac_horizon):
            agent_ac = ac[:, ac_ix] if len(ac.shape) > 2 else ac
            agent_ac = np.array(agent_ac, copy=True)
            if dof > 2:
                close_mask = agent_ac[:, 0] > 0.9
                open_mask = agent_ac[:, 0] < 0.1
                grip_state[close_mask] = 1.0
                grip_state[open_mask] = 0.0
                agent_ac[:, 0] = grip_state
            env.step_async(agent_ac, dummy_reward=True)
            state, _, done, _ = env.step_wait()
            rgb_render = render = env.env_method("render")
            obs = organize_obs(render, rgb_render, state)
            obs_history.append(obs)
            if len(obs) > obs_horizon:
                obs_history = obs_history[-obs_horizon:]
            for i in range(num_envs):
                images[i].append(rgb_render[i]["images"][0][..., :3])
        t += 1
        pbar.update(1)
    pbar.close()
    rews = np.array(env.env_method("compute_reward"))
    print(f"Episode rewards: {rews.round(3)}.")

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        for ep_ix in range(num_envs):
            sp_list = history[ep_ix]["selector_params"]
            valid = [p for p in sp_list if p is not None]
            if valid:
                param_dim = valid[0].shape[-1]
                sp_arr = np.stack([
                    p if p is not None else np.full(param_dim, np.nan)
                    for p in sp_list
                ])
            else:
                sp_arr = np.array([])
            np.savez(
                os.path.join(
                    log_dir, f"eval_{ckpt_name}_ep{ep_ix:02d}_rew{rews[ep_ix]:.3f}.npz"
                ),
                action=np.array(history[ep_ix]["action"]),
                eef_pos=np.array(history[ep_ix]["eef_pos"]),
                entropy=np.array(history[ep_ix]["entropy"]),
                selector_params=sp_arr,
            )
        if use_wandb:
            for ep_ix in range(min(4, num_envs)):
                trace = np.array(history[ep_ix]["entropy"])
                if np.any(np.isfinite(trace)):
                    table = wandb.Table(
                        data=[[t, float(h)] for t, h in enumerate(trace) if np.isfinite(h)],
                        columns=["timestep", "entropy"],
                    )
                    wandb.log({
                        f"eval/entropy_trace_ep{ep_ix}": wandb.plot.line(
                            table, "timestep", "entropy",
                            title=f"H(p_phi) trace - ep{ep_ix} rew={rews[ep_ix]:.2f}",
                        )
                    })

    images = np.array(images)
    metrics = dict(rew=np.mean(rews))
    if vis:
        vis_frames = images
        vis_rews = np.zeros_like(vis_frames[:, :, :20])
        for i in range(num_envs):
            num_pixels = int(vis_frames.shape[-2] * rews[i])
            vis_rews[i, :, 2:, :num_pixels] = 255
        vis_frames = np.concatenate([vis_frames, vis_rews], axis=2)
        if use_wandb:
            metrics["vis_pc"] = wandb.Object3D(sample_pc)
        metrics["rew_values"] = rews
        metrics["vis_rollout"] = vis_frames
        metrics["mean_pc_size"] = mean_num_points_in_pc
    return metrics
