import os
import sys
import copy
import hydra
import torch
import wandb
import omegaconf
import numpy as np
import getpass as gt
from tqdm import tqdm
from glob import glob
from omegaconf import OmegaConf

from pefm.utils.media import save_video
from pefm.utils.misc import get_env_class, get_dataset, get_agent
from pefm.vec_eval import run_eval
from pefm_envs.subproc_vec_env import SubprocVecEnv


@hydra.main(config_path="configs", config_name="base_pefm", version_base="1.1")
def main(cfg):
    assert cfg.mode == "train"
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    batch_size = cfg.training.batch_size

    if cfg.use_wandb:
        wandb_config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=False
        )
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            tags=["train"],
            name=cfg.prefix,
            settings=wandb.Settings(code_dir="."),
            config=wandb_config,
        )
    from hydra.core.hydra_config import HydraConfig
    log_dir = HydraConfig.get().runtime.output_dir
    os.makedirs(log_dir, exist_ok=True)

    train_dataset = get_dataset(cfg, "train")
    num_workers = cfg.data.dataset.num_workers
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.seed)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        generator=loader_generator,
    )
    cfg.data.dataset.num_training_steps = (
        cfg.training.num_epochs * len(train_dataset) // batch_size
    )

    env_fns = []
    env_class = get_env_class(cfg.env.env_class)
    env_args = dict(OmegaConf.to_container(cfg.env.args, resolve=True))

    def create_env(env_args, i):
        env_args["seed"] = cfg.seed * 100 + i
        return env_class(OmegaConf.create(env_args))

    if cfg.training.eval_interval <= cfg.training.num_epochs:
        env = SubprocVecEnv(
            [
                lambda seed=i: create_env(env_args, seed)
                for i in range(cfg.training.num_eval_episodes)
            ]
        )
    else:
        env = None

    agent = get_agent(cfg.agent.agent_name)(cfg)
    if cfg.training.ckpt is not None:
        agent.load_snapshot(cfg.training.ckpt)
        start_epoch_ix = int(cfg.training.ckpt.split("/")[-1].split(".")[0][4:])
    else:
        start_epoch_ix = 0

    global_step = 0
    for epoch_ix in tqdm(range(start_epoch_ix, cfg.training.num_epochs)):
        batch_ix = 0
        for batch in tqdm(train_loader, leave=False, desc="Batches"):
            train_metrics = agent.update(
                batch, vis=epoch_ix % cfg.training.vis_interval == 0 and batch_ix == 0
            )
            if cfg.use_wandb:
                wandb.log(
                    {"train/" + k: v for k, v in train_metrics.items()},
                    step=global_step,
                )
                wandb.log({"epoch": epoch_ix}, step=global_step)
            del train_metrics
            global_step += 1
            batch_ix += 1
        if (
            (
                epoch_ix % cfg.training.eval_interval == 0
                or epoch_ix == cfg.training.num_epochs - 1
            )
            and epoch_ix > 0
            and env is not None
        ):
            eval_metrics = run_eval(
                env,
                agent,
                vis=True,
                num_episodes=cfg.training.num_eval_episodes,
                reduce_horizon_dim=cfg.data.dataset.reduce_horizon_dim,
                use_wandb=cfg.use_wandb,
            )
            if cfg.use_wandb:
                if epoch_ix > cfg.training.eval_interval and "vis_pc" in eval_metrics:
                    del eval_metrics["vis_pc"]
                wandb.log(
                    {
                        "eval/" + k: v
                        for k, v in eval_metrics.items()
                        if not k in ["vis_rollout", "rew_values"]
                    },
                    step=global_step,
                )
                if "vis_rollout" in eval_metrics:
                    for eval_idx, eval_video in enumerate(eval_metrics["vis_rollout"]):
                        video_path = os.path.join(
                            log_dir,
                            f"eval{epoch_ix:05d}_ep{eval_idx}_rew{eval_metrics['rew_values'][eval_idx]}.mp4",
                        )
                        save_video(eval_video, video_path)
                        print(f"Saved eval video to {video_path}")
            del eval_metrics
        if (
            epoch_ix % cfg.training.save_interval == 0
            or epoch_ix == cfg.training.num_epochs - 1
        ):
            save_path = os.path.join(log_dir, f"ckpt{epoch_ix:05d}.pth")
            num_ckpt_to_keep = 10
            if len(list(glob(os.path.join(log_dir, "ckpt*.pth")))) > num_ckpt_to_keep:
                for fn in list(sorted(glob(os.path.join(log_dir, "ckpt*.pth"))))[
                    :-num_ckpt_to_keep
                ]:
                    os.remove(fn)
            agent.save_snapshot(save_path)


if __name__ == "__main__":
    main()
