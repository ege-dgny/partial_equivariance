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
        grip_state = np.array(obs["state"])[..., -1].reshape(num_envs, -1)[:, -1].astype(np.float32)

    sample_pc = render[0]["pc"]
    mean_num_points_in_pc = np.mean([len(render[k]["pc"]) for k in range(len(render))])

    done = [False] * num_envs
    if log_dir is not None:
        history = []
        for i in range(num_envs):
            history.append(
                dict(action=[], eef_pos=[], entropy=[], selector_params=[])
            )
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

        # Request per-step selector entropy + params (Phase 1.4).
        # ac_dict is a list[dict|None] of length num_envs; None for invalid PCs.
        ac, ac_dict = agent.act(agent_obs, return_dict=True)

        if log_dir is not None:
            for i in range(num_envs):
                history[i]["action"].append(ac[i])
                history[i]["eef_pos"].append(obs["state"][i])
                entry = ac_dict[i] if i < len(ac_dict) else None
                if entry is not None and "selector_entropy" in entry:
                    history[i]["entropy"].append(float(entry["selector_entropy"]))
                    history[i]["selector_params"].append(
                        np.asarray(entry["selector_params"])
                    )
                else:
                    history[i]["entropy"].append(float("nan"))
                    # Use None as placeholder; reconciled at save time.
                    history[i]["selector_params"].append(None)

        for ac_ix in range(ac_horizon):
            agent_ac = ac[:, ac_ix] if len(ac.shape) > 2 else ac
            agent_ac = np.array(agent_ac, copy=True)
            if dof > 2:
                # Sticky/latching gripper (restored from 8e3d31f): close >0.9, open <0.1,
                # else hold. Pick-and-place needs the grasp held through transport; the raw
                # fluctuating gripper command drops the object (regressed Can 0.6 -> 0.05).
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
            # Reconcile selector_params: fill None placeholders with NaN arrays
            # of the same dim as observed entries (if any), else skip.
            sp_list = history[ep_ix]["selector_params"]
            valid = [p for p in sp_list if p is not None]
            if valid:
                param_dim = valid[0].shape[-1]
                sp_arr = np.stack(
                    [
                        p if p is not None else np.full(param_dim, np.nan)
                        for p in sp_list
                    ]
                )
            else:
                sp_arr = np.array([])
            np.savez(
                os.path.join(
                    log_dir, f"eval_{ckpt_name}_ep{ep_ix:02d}_rew{rews[ep_ix]:.3f}.npz"
                ),
                action=np.array(history[ep_ix]["action"]),
                eef_pos=np.array(history[ep_ix]["eef_pos"]),
                entropy=np.array(history[ep_ix]["entropy"]),       # (T,)
                selector_params=sp_arr,                              # (T, param_dim)
            )

    images = np.array(images)
    metrics = dict(
        rew=np.mean(rews),
        success_rate=np.mean(rews >= 0.5),
        rew_std=np.std(rews),
    )

    # Phase 1.5: W&B line plot of entropy trace for the first 4 episodes.
    # This is the single most important figure for the partial-equivariance
    # claim: if H(t) is flat the selector is not observation-conditioned.
    if use_wandb and log_dir is not None:
        for ep_ix in range(min(num_envs, 4)):
            trace = np.array(history[ep_ix]["entropy"]).squeeze()
            if trace.ndim == 0 or trace.size == 0:
                continue
            table = wandb.Table(
                data=[[int(t), float(h)] for t, h in enumerate(trace)],
                columns=["timestep", "entropy"],
            )
            metrics[f"entropy_trace_ep{ep_ix}"] = wandb.plot.line(
                table,
                "timestep",
                "entropy",
                title=f"H(p_phi) trace - ep{ep_ix} rew={rews[ep_ix]:.2f}",
            )

    if use_wandb:
        # Per-episode reward table — one row per episode, logged at every eval.
        rew_table = wandb.Table(
            data=[[int(i), float(rews[i]), int(rews[i] >= 0.5)] for i in range(len(rews))],
            columns=["episode", "reward", "success"],
        )
        metrics["reward_table"] = rew_table

        # C4 selector distribution chart: aggregate selector_params across all
        # episodes, compute softmax, log per-bin mean probabilities as a bar chart.
        all_params = []
        for ep_ix in range(num_envs):
            for p in history[ep_ix]["selector_params"]:
                if p is not None:
                    all_params.append(p)
        if all_params:
            params_arr = np.stack(all_params)           # (T_total, param_dim)
            exp_p = np.exp(params_arr - params_arr.max(axis=-1, keepdims=True))
            probs_arr = exp_p / exp_p.sum(axis=-1, keepdims=True)  # (T_total, K)
            bin_means = probs_arr.mean(axis=0)           # (K,)
            bin_table = wandb.Table(
                data=[[f"bin{b}", float(bin_means[b])] for b in range(len(bin_means))],
                columns=["bin", "mean_prob"],
            )
            metrics["selector_distribution"] = wandb.plot.bar(
                bin_table, "bin", "mean_prob",
                title="Selector distribution (uniform=symmetry preserved)",
            )

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
