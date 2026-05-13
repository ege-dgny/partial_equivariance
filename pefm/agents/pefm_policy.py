"""
Partially Equivariant Flow Matching Policy.

Composes:
1. Perception Encoder (PointNet): O -> z
2. Symmetry Selector (MLP): z -> p_phi(g|o)
3. Base Vector Field (ConditionalUnet1D): v_base(t, x_t, o)
4. ODE Solver: integration for inference

Core equation:
    v_PE(t, x, o) = E_{g~p_phi(.|o)} [rho_out(g^-1) . v_base(t, rho_in(g).x, rho_in(g).o)]
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from pefm.vision.pointnet_encoder import PointNetEncoder
from pefm.networks.conditional_unet1d import ConditionalUnet1D
from pefm.networks.resnet_with_gn import get_resnet, replace_bn_with_gn
from pefm.utils.ema_model import EMAModel
from pefm.symmetry.groups import get_group
from pefm.symmetry.distributions import get_distribution
from pefm.symmetry.selector import SymmetrySelector
from pefm.flow_matching.ot_sampler import OTConditionalFlowMatching
from pefm.flow_matching.ode_solver import ODESolver


class PEFMPolicy(nn.Module):
    def __init__(self, cfg, device="cpu"):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = hidden_dim = cfg.model.hidden_dim
        self.obs_mode = cfg.model.obs_mode
        self.device = device

        self.pred_horizon = cfg.model.pred_horizon
        self.obs_horizon = cfg.model.obs_horizon
        self.action_horizon = cfg.model.ac_horizon

        self.num_eef = cfg.env.num_eef
        self.eef_dim = cfg.env.eef_dim
        self.dof = cfg.env.dof
        self.canonicalize = cfg.model.get("canonicalize", False)

        if cfg.model.obs_mode == "state":
            self.obs_dim = self.num_eef * self.eef_dim
        elif cfg.model.obs_mode == "rgb":
            self.obs_dim = 512 + self.num_eef * self.eef_dim
        else:
            self.obs_dim = hidden_dim + self.num_eef * self.eef_dim
        self.action_dim = self.dof * cfg.env.num_eef

        # 1. Perception Encoder
        if self.obs_mode.startswith("pc"):
            self.encoder = PointNetEncoder(
                h_dim=hidden_dim,
                c_dim=hidden_dim,
                num_layers=cfg.model.encoder.backbone_args.num_layers,
            )
        elif self.obs_mode == "rgb":
            self.encoder = replace_bn_with_gn(get_resnet("resnet18"))
        else:
            self.encoder = nn.Identity()

        # 2. Symmetry Selector
        self.group = get_group(cfg.model.symmetry.group_type)
        self.distribution = get_distribution(cfg.model.symmetry)
        obs_cond_dim = self.obs_dim * self.obs_horizon
        self.selector = SymmetrySelector(
            z_dim=obs_cond_dim,
            hidden_dim=cfg.model.symmetry.hidden_dim,
            group=self.group,
            distribution=self.distribution,
        )

        # 3. Base Vector Field (non-equivariant ConditionalUnet1D)
        self.velocity_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            diffusion_step_embed_dim=obs_cond_dim,
            global_cond_dim=obs_cond_dim,
        )

        # 4. Module dict for EMA
        self.nets = nn.ModuleDict(
            {
                "encoder": self.encoder,
                "selector": self.selector,
                "velocity_net": self.velocity_net,
            }
        )
        self.ema = EMAModel(model=copy.deepcopy(self.nets), power=0.75)

        # 5. Flow matching
        sigma_min = cfg.model.get("flow_sigma_min", 0.001)
        self.ot_cfm = OTConditionalFlowMatching(sigma_min=sigma_min)

        # 6. ODE solver (inference)
        self.ode_solver = ODESolver(
            num_steps=cfg.model.ode_steps,
            method=cfg.model.ode_method,
        )

        # 7. PEFM hyperparameters
        self.num_group_samples = cfg.model.symmetry.num_samples
        self.num_group_samples_eval = cfg.model.symmetry.get(
            "num_samples_eval", cfg.model.symmetry.num_samples
        )
        self.entropy_weight = cfg.model.symmetry.entropy_weight

        self._init_torch_compile()

        num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Initialized PEFM Policy with {num_parameters} parameters")

    def _init_torch_compile(self):
        if self.cfg.model.use_torch_compile:
            self.encoder_handle = torch.compile(self.encoder)
            self.velocity_net_handle = torch.compile(self.velocity_net)

    def step_ema(self):
        self.ema.step(self.nets)

    def _encode_obs(self, pc, state, encoder_module):
        """
        Encode observation to conditioning vector.

        pc: (B, obs_horizon, num_pts, 3)
        state: (B, obs_horizon, state_dim)
        encoder_module: encoder to use (training nets or EMA)

        Returns: obs_cond (B, obs_horizon * obs_dim)
        """
        pc_shape = pc.shape
        batch_size = pc.shape[0]

        if self.obs_mode == "state":
            z = state
        else:
            flattened_pc = pc.reshape(
                batch_size * self.obs_horizon, *pc_shape[-2:]
            )
            z = encoder_module(flattened_pc.permute(0, 2, 1))["global"]
            z = z.reshape(batch_size, self.obs_horizon, -1)
            z = torch.cat([z, state], dim=-1)

        obs_cond = z.reshape(batch_size, -1)
        return obs_cond

    def _canonicalize_obs(self, pc, state):
        """
        Center and scale-normalize point clouds and states for
        translation/scale invariance. Uses LAST obs frame's stats.

        pc: (B, obs_h, P, 3)
        state: (B, obs_h, num_eef * eef_dim)
        Returns: pc_canon, state_canon, centroid (B, 1, 3), scale (B,)
        """
        B = pc.shape[0]
        pc_last = pc[:, -1]                                    # (B, P, 3)
        centroid = pc_last.mean(dim=1, keepdim=True)           # (B, 1, 3)

        # Center all frames by last frame's centroid
        pc_canon = pc - centroid[:, None]                      # (B, obs_h, P, 3)

        # Scale = mean point norm in centered last frame
        scale = pc_canon[:, -1].norm(dim=-1).mean(dim=-1)     # (B,)
        scale = scale.clamp(min=1e-6)
        pc_canon = pc_canon / scale[:, None, None, None]

        # Canonicalize state position (indices 0:3 per EEF only)
        state_canon = state.clone()
        s = state_canon.view(B, self.obs_horizon, self.num_eef, self.eef_dim)
        s[..., :3] = (s[..., :3] - centroid[:, None]) / scale[:, None, None, None]
        state_canon = s.view(B, self.obs_horizon, -1)

        return pc_canon, state_canon, centroid, scale

    def _canonicalize_action(self, action, centroid, scale):
        """
        Canonicalize ground truth action for training.
        Only position deltas are scaled. Rotation deltas and gripper unchanged.
        """
        B = action.shape[0]
        ac = action.clone()
        a = ac.view(B, self.pred_horizon, self.num_eef, self.dof)
        s = scale[:, None, None, None]

        if self.dof >= 4:
            # dof=4: [gripper, dx, dy, dz], dof=7: [gripper, dx, dy, dz, drx, dry, drz]
            if self.cfg.model.ac_mode == "abs":
                a[..., 1:4] = (a[..., 1:4] - centroid[:, None]) / s
            else:
                a[..., 1:4] = a[..., 1:4] / s
        elif self.dof == 3:
            if self.cfg.model.ac_mode == "abs":
                a[..., :3] = (a[..., :3] - centroid[:, None]) / s
            else:
                a[..., :3] = a[..., :3] / s

        return ac.view(B, self.pred_horizon, -1)

    def _uncanonicalize_action(self, action, centroid, scale):
        """Inverse of _canonicalize_action for inference."""
        B = action.shape[0]
        ac = action.clone()
        a = ac.view(B, self.pred_horizon, self.num_eef, self.dof)
        s = scale[:, None, None, None]

        if self.dof >= 4:
            if self.cfg.model.ac_mode == "abs":
                a[..., 1:4] = a[..., 1:4] * s + centroid[:, None]
            else:
                a[..., 1:4] = a[..., 1:4] * s
        elif self.dof == 3:
            if self.cfg.model.ac_mode == "abs":
                a[..., :3] = a[..., :3] * s + centroid[:, None]
            else:
                a[..., :3] = a[..., :3] * s

        return ac.view(B, self.pred_horizon, -1)

    def _pefm_velocity_batched(
        self, x_t, t_flat, obs_cond, g_samples, pc, state, encoder_module, vel_module,
        return_individual=False
    ):
        """
        Compute PEFM averaged velocity using batched forward pass.

        Transforms inputs for all N group samples, runs encoder and velocity_net
        once with batch size B*N, then inverse-transforms and averages.

        x_t: (B, H, D) noisy actions
        t_flat: (B,) time
        obs_cond: (B, obs_cond_dim) original obs conditioning (for selector)
        g_samples: (B, N) group elements
        pc: (B, obs_horizon, num_pts, 3) raw point clouds
        state: (B, obs_horizon, state_dim) raw states
        encoder_module: encoder to use
        vel_module: velocity_net to use
        return_individual: if True also return (B, N, H, D) per-sample velocities

        Returns: v_pe (B, H, D) averaged velocity [, v_global (B, N, H, D)]
        """
        B = x_t.shape[0]
        N = g_samples.shape[1]

        # Expand inputs for N group samples: (B, N, ...)
        pc_exp = pc.unsqueeze(1).expand(-1, N, *pc.shape[1:])  # (B, N, obs_h, P, 3)
        state_exp = state.unsqueeze(1).expand(
            -1, N, *state.shape[1:]
        )  # (B, N, obs_h, S)
        xt_exp = x_t.unsqueeze(1).expand(-1, N, *x_t.shape[1:])  # (B, N, H, D)
        t_exp = t_flat.unsqueeze(1).expand(-1, N)  # (B, N)

        # Apply group transforms
        g_flat = g_samples.reshape(B * N)

        # Transform point clouds: reshape to (B*N, obs_h, P, 3)
        pc_flat = pc_exp.reshape(B * N, *pc.shape[1:])
        # Rotate each obs_horizon frame
        pc_rot_frames = []
        for h_idx in range(self.obs_horizon):
            pc_frame = pc_flat[:, h_idx]  # (B*N, P, 3)
            pc_frame_rot = self.group.transform_points(g_flat, pc_frame)
            pc_rot_frames.append(pc_frame_rot)
        pc_rot = torch.stack(pc_rot_frames, dim=1)  # (B*N, obs_h, P, 3)

        # Transform states
        state_flat = state_exp.reshape(B * N, *state.shape[1:])
        state_rot_frames = []
        for h_idx in range(self.obs_horizon):
            s_frame = state_flat[:, h_idx]  # (B*N, state_dim)
            s_frame_rot = self.group.transform_state(
                g_flat, s_frame.unsqueeze(1),
                num_eef=self.num_eef, eef_dim=self.eef_dim
            ).squeeze(1)
            state_rot_frames.append(s_frame_rot)
        state_rot = torch.stack(state_rot_frames, dim=1)  # (B*N, obs_h, S)

        # Transform actions
        xt_flat = xt_exp.reshape(B * N, *x_t.shape[1:])  # (B*N, H, D)
        xt_rot = self.group.transform_action(
            g_flat, xt_flat, dof=self.dof, num_eef=self.num_eef
        )

        # Encode rotated observations
        obs_cond_rot = self._encode_obs(pc_rot, state_rot, encoder_module)

        # Predict velocity
        t_flat_bn = t_exp.reshape(B * N)
        # Scale t to match sinusoidal embedding range (flow t in [0,1] -> [0, 1000])
        t_for_embed = (t_flat_bn * 1000).long()

        v_prime = vel_module(
            sample=xt_rot, timestep=t_for_embed, global_cond=obs_cond_rot
        )  # (B*N, H, D)

        # Inverse-transform velocities
        v_prime_reshaped = v_prime.reshape(B, N, *x_t.shape[1:])  # (B, N, H, D)
        v_global_list = []
        for i in range(N):
            g_i = g_samples[:, i]  # (B,)
            v_i = v_prime_reshaped[:, i]  # (B, H, D)
            v_global_i = self.group.inverse_transform_action(
                g_i, v_i, dof=self.dof, num_eef=self.num_eef
            )
            v_global_list.append(v_global_i)
        v_global = torch.stack(v_global_list, dim=1)  # (B, N, H, D)

        # Average over group samples
        v_pe = v_global.mean(dim=1)  # (B, H, D)

        if return_individual:
            return v_pe, v_global
        return v_pe

    def compute_loss(self, pc, state, gt_action):
        """
        Compute PEFM flow matching loss with entropy regularization.

        L_total = L_flow - lambda * H(p_phi)

        pc: (B, obs_horizon, num_pts, 3)
        state: (B, obs_horizon, state_dim)
        gt_action: (B, pred_horizon, action_dim)

        Returns: loss_total, metrics_dict
        """
        batch_size = gt_action.shape[0]

        # 0. Canonicalize for scale/translation invariance
        if self.canonicalize:
            pc, state, canon_centroid, canon_scale = self._canonicalize_obs(pc, state)
            gt_action = self._canonicalize_action(gt_action, canon_centroid, canon_scale)

        # 1. Encode original observation (for selector)
        obs_cond = self._encode_obs(pc, state, self.encoder)

        # 2. Sample group elements from selector
        g_samples, entropy = self.selector.sample_and_entropy(
            obs_cond, self.num_group_samples
        )

        # 3. Sample flow matching time and noise
        t = torch.rand(batch_size, 1, 1, device=gt_action.device)
        x0 = torch.randn_like(gt_action)
        x1 = gt_action

        x_t = self.ot_cfm.sample_xt(x0, x1, t)
        u_t = self.ot_cfm.target_velocity(x0, x1)

        # 4. PEFM averaged velocity prediction (batched)
        t_flat = t.squeeze(-1).squeeze(-1)  # (B,)
        v_pe, v_individual = self._pefm_velocity_batched(
            x_t, t_flat, obs_cond, g_samples,
            pc, state, self.encoder, self.velocity_net,
            return_individual=True,
        )

        # 5. Flow matching loss on individual per-g predictions.
        # ||mean_i(v_i) - u_t||^2 only constrains the average, allowing
        # non-equivariant individual predictions that cancel on average but
        # give inconsistent ODE trajectories at eval. Using per-sample loss
        # mean_i(||v_i - u_t||^2) directly enforces equivariance per g.
        u_t_exp = u_t.unsqueeze(1).expand_as(v_individual)
        loss_flow = F.mse_loss(v_individual, u_t_exp)

        # 6. Entropy regularization: -lambda * H encourages high entropy
        loss_entropy = -self.entropy_weight * entropy.mean()

        # 7. Total loss
        loss_total = loss_flow + loss_entropy

        metrics = {
            "loss_flow": loss_flow.item(),
            "loss_entropy": loss_entropy.item(),
            "entropy": entropy.mean().item(),
            "loss_total": loss_total.item(),
        }
        if self.canonicalize:
            metrics["canon_scale_mean"] = canon_scale.mean().item()

        return loss_total, metrics

    def forward(self, obs, debug=False):
        """
        Inference: ODE integration with PEFM averaged velocity.

        obs: dict with 'pc' (B, obs_horizon, P, 3) and 'state' (B, obs_horizon, S)
        Returns: dict with 'ac' (B, pred_horizon, action_dim)
        """
        pc = obs["pc"]
        state = obs["state"]

        if self.obs_mode.startswith("pc"):
            pc = self.pc_normalizer.normalize(pc)
        state = self.state_normalizer.normalize(state)

        # Canonicalize if enabled
        canon_params = None
        if self.canonicalize:
            pc, state, canon_centroid, canon_scale = self._canonicalize_obs(pc, state)
            canon_params = (canon_centroid, canon_scale)

        batch_size = pc.shape[0]
        ema_nets = self.ema.averaged_model

        N = self.num_group_samples_eval

        # 1. Encode original observation (for selector)
        obs_cond = self._encode_obs(pc, state, ema_nets["encoder"])

        # 2. Sample group elements once; reuse fixed throughout ODE so the
        # trajectory is deterministic given x_0 and g_samples (consistent frame).
        with torch.no_grad():
            g_samples, _ = ema_nets[selector].sample_and_entropy(obs_cond, N)

        # 3. Initialize noise
        initial_noise_scale = 0.0 if debug else 1.0
        x0 = (
            torch.randn(batch_size, self.pred_horizon, self.action_dim).to(self.device)
            * initial_noise_scale
        )

        # 4. ODE integration with PEFM velocity
        def pefm_velocity(x_t, t):
            return self._pefm_velocity_batched(
                x_t, t, obs_cond, g_samples,
                pc, state, ema_nets[encoder], ema_nets[velocity_net]
            )

        predicted_actions = self.ode_solver.solve(pefm_velocity, x0)

        # Un-canonicalize predicted actions
        if canon_params is not None:
            predicted_actions = self._uncanonicalize_action(
                predicted_actions, *canon_params
            )

        return dict(ac=predicted_actions)
