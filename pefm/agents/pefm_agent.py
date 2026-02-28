"""
PEFM Agent: Training loop and inference for Partially Equivariant Flow Matching.

Follows the DPAgent pattern from EquiBot:
- Lazy normalizer initialization from first batch
- EMA model for inference
- Invalid point cloud handling
- Checkpoint management with normalizer states
"""

import numpy as np
import torch
from torch import nn

from pefm.utils.norm import Normalizer, RotationAwareNormalizer
from pefm.utils.misc import to_torch
from pefm.agents.pefm_policy import PEFMPolicy
from pefm.utils.lr_scheduler import get_scheduler


class PEFMAgent(object):
    def __init__(self, cfg):
        print(f"Initializing PEFM agent.")
        self.cfg = cfg
        self._init_actor()
        if cfg.mode == "train":
            self.optimizer = torch.optim.AdamW(
                self.actor.nets.parameters(),
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay,
            )
            self.lr_scheduler = get_scheduler(
                name="cosine",
                optimizer=self.optimizer,
                num_warmup_steps=500,
                num_training_steps=cfg.data.dataset.num_training_steps,
            )
        self.device = cfg.device
        self.num_eef = cfg.env.num_eef
        self.dof = cfg.env.dof
        self.num_points = cfg.data.dataset.num_points
        self.obs_mode = cfg.model.obs_mode
        self.ac_mode = cfg.model.ac_mode
        self.obs_horizon = cfg.model.obs_horizon
        self.pred_horizon = cfg.model.pred_horizon
        self.ac_horizon = cfg.model.ac_horizon
        self.shuffle_pc = cfg.data.dataset.shuffle_pc

        self.pc_normalizer = None
        self.state_normalizer = None
        self.ac_normalizer = None

    def _init_actor(self):
        self.actor = PEFMPolicy(self.cfg, device=self.cfg.device).to(self.cfg.device)
        self.actor.ema.averaged_model.to(self.cfg.device)

    def _init_normalizers(self, batch):
        use_rot_aware = self.cfg.model.get("rotation_aware_norm", False)

        if self.obs_mode.startswith("pc") and self.pc_normalizer is None:
            flattened_pc = batch["pc"].view(-1, 3)
            if use_rot_aware:
                self.pc_normalizer = RotationAwareNormalizer(
                    flattened_pc, coupled_groups=[[0, 1]]
                )
            else:
                self.pc_normalizer = Normalizer(flattened_pc)
            self.actor.pc_normalizer = self.pc_normalizer
            print(f"PC normalization stats: {self.pc_normalizer.stats}")

        if self.state_normalizer is None:
            state = batch["eef_pos"]
            flattened_state = state.view(-1, state.shape[-1])
            if use_rot_aware:
                eef_dim = self.cfg.env.eef_dim
                state_coupled = []
                for eef_idx in range(self.num_eef):
                    off = eef_idx * eef_dim
                    state_coupled.append([off + 0, off + 1])    # pos XY
                    state_coupled.append([off + 3, off + 4])    # xdir XY
                    state_coupled.append([off + 6, off + 7])    # zdir XY
                    state_coupled.append([off + 9, off + 10])   # grav XY
                self.state_normalizer = RotationAwareNormalizer(
                    flattened_state, coupled_groups=state_coupled
                )
            else:
                self.state_normalizer = Normalizer(flattened_state)
            self.actor.state_normalizer = self.state_normalizer
            print(f"State normalization stats: {self.state_normalizer.stats}")

        if self.ac_normalizer is None:
            gt_action = batch["action"]
            flattened_gt_action = gt_action.view(-1, gt_action.shape[-1])
            if use_rot_aware:
                ac_coupled = []
                for eef_idx in range(self.num_eef):
                    off = eef_idx * self.dof
                    ac_coupled.append([off + 1, off + 2])       # vel XY
                    if self.dof >= 7:
                        ac_coupled.append([off + 4, off + 5])   # rot-vel XY
                self.ac_normalizer = RotationAwareNormalizer(
                    flattened_gt_action, coupled_groups=ac_coupled
                )
            else:
                self.ac_normalizer = Normalizer(flattened_gt_action)
            print(f"Action normalization stats: {self.ac_normalizer.stats}")

    def train(self, training=True):
        self.actor.nets.train(training)

    def act(self, obs, return_dict=False, debug=False):
        """
        Inference: process observations, run PEFM ODE integration, return actions.

        Follows DPAgent.act() pattern:
        - Handles batched/unbatched input
        - Skips invalid point clouds
        - Uses EMA model
        """
        self.train(False)
        assert isinstance(obs["pc"][0][0], np.ndarray)
        if len(obs["state"].shape) == 3:
            assert len(obs["pc"][0].shape) == 2
            obs["pc"] = [[x] for x in obs["pc"]]
            for k in obs:
                if k != "pc" and isinstance(obs[k], np.ndarray):
                    obs[k] = obs[k][:, None]
            has_batch_dim = False
        elif len(obs["state"].shape) == 4:
            assert len(obs["pc"][0][0].shape) == 2
            has_batch_dim = True
        else:
            raise ValueError("Input format not recognized.")

        ac_dim = self.num_eef * self.dof
        batch_size = len(obs["pc"][0])

        state = obs["state"].reshape(tuple(obs["state"].shape[:2]) + (-1,))

        xyzs = []
        ac = np.zeros([batch_size, self.pred_horizon, ac_dim])
        if return_dict:
            ac_dict = []
            for i in range(batch_size):
                ac_dict.append(None)
        forward_idxs = list(np.arange(batch_size))
        for pcs in obs["pc"]:
            xyzs.append([])
            for batch_idx, xyz in enumerate(pcs):
                if not batch_idx in forward_idxs:
                    xyzs[-1].append(np.zeros((self.num_points, 3)))
                elif xyz.shape[0] == 0:
                    forward_idxs.remove(batch_idx)
                    xyzs[-1].append(np.zeros((self.num_points, 3)))
                elif self.shuffle_pc:
                    choice = np.random.choice(
                        xyz.shape[0], self.num_points, replace=True
                    )
                    xyz = xyz[choice, :]
                    xyzs[-1].append(xyz)
                else:
                    step = xyz.shape[0] // self.num_points
                    xyz = xyz[::step, :][: self.num_points]
                    xyzs[-1].append(xyz)

        if len(forward_idxs) > 0:
            torch_obs = dict(
                pc=torch.tensor(np.array(xyzs).swapaxes(0, 1)[forward_idxs])
                .to(self.device)
                .float(),
                state=torch.tensor(state.swapaxes(0, 1)[forward_idxs])
                .to(self.device)
                .float(),
            )
            for k in obs:
                if not k in ["pc", "state"] and isinstance(obs[k], np.ndarray):
                    torch_obs[k] = (
                        torch.tensor(obs[k].swapaxes(0, 1)[forward_idxs])
                        .to(self.device)
                        .float()
                    )
            with torch.no_grad():
                raw_ac_dict = self.actor(torch_obs, debug=debug)
        else:
            raw_ac_dict = dict(
                ac=torch.zeros(
                    (batch_size, self.actor.pred_horizon, self.actor.action_dim)
                ).to(self.device)
            )
        for i, idx in enumerate(forward_idxs):
            if return_dict:
                ac_dict[idx] = {k: v[i] for k, v in raw_ac_dict.items()}
            unnormed_action = (
                self.ac_normalizer.unnormalize(raw_ac_dict["ac"][i])
                .detach()
                .cpu()
                .numpy()
            )
            ac[idx] = unnormed_action
        if not has_batch_dim:
            ac = ac[0]
            if return_dict:
                ac_dict = ac_dict[0]
        if return_dict:
            return ac, ac_dict
        else:
            return ac

    def update(self, batch, vis=False):
        """
        Training step: PEFM flow matching with entropy regularization.

        L_total = L_flow - lambda * H(p_phi)
        """
        self.train()

        batch = to_torch(batch, self.device)
        batch["eef_pos"] = batch["eef_pos"].reshape(
            tuple(batch["eef_pos"].shape[:2]) + (-1,)
        )
        pc = batch["pc"]
        state = batch["eef_pos"]
        gt_action = batch["action"]

        if self.state_normalizer is None or self.ac_normalizer is None:
            self._init_normalizers(batch)
        if self.obs_mode.startswith("pc"):
            pc = self.pc_normalizer.normalize(pc)
        state = self.state_normalizer.normalize(state)
        gt_action = self.ac_normalizer.normalize(gt_action)

        # Compute PEFM loss
        loss, metrics = self.actor.compute_loss(pc, state, gt_action)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.nets.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.lr_scheduler.step()

        # EMA update
        self.actor.step_ema()

        # Anneal Gumbel temperature if using discrete groups
        if hasattr(self.actor.distribution, "anneal_tau"):
            anneal_rate = self.cfg.model.symmetry.get("gumbel_anneal_rate", 0.0003)
            self.actor.distribution.anneal_tau(rate=anneal_rate)

        return metrics

    def save_snapshot(self, save_path):
        state_dict = dict(
            actor=self.actor.state_dict(),
            ema_model=self.actor.ema.averaged_model.state_dict(),
            pc_normalizer=self.pc_normalizer.state_dict(),
            state_normalizer=self.state_normalizer.state_dict(),
            ac_normalizer=self.ac_normalizer.state_dict(),
        )
        torch.save(state_dict, save_path)

    def _fix_state_dict_keys(self, state_dict):
        return {k: v for k, v in state_dict.items() if not "handle" in k}

    @staticmethod
    def _make_normalizer(saved_stats):
        """Auto-detect normalizer type from checkpoint keys."""
        if "center" in saved_stats:
            return RotationAwareNormalizer(saved_stats)
        return Normalizer(saved_stats)

    def load_snapshot(self, save_path):
        state_dict = torch.load(save_path, map_location=self.device)
        self.state_normalizer = self._make_normalizer(state_dict["state_normalizer"])
        self.actor.state_normalizer = self.state_normalizer
        self.ac_normalizer = self._make_normalizer(state_dict["ac_normalizer"])
        if self.obs_mode.startswith("pc"):
            self.pc_normalizer = self._make_normalizer(state_dict["pc_normalizer"])
            self.actor.pc_normalizer = self.pc_normalizer
        if hasattr(self, "encoder_handle"):
            del self.encoder_handle
            del self.velocity_net_handle
        self.actor.load_state_dict(
            self._fix_state_dict_keys(state_dict["actor"])
        )
        self.actor._init_torch_compile()
        self.actor.ema.averaged_model.load_state_dict(state_dict["ema_model"])
