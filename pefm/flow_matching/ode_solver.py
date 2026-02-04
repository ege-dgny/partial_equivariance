"""
ODE solver for flow matching inference.

Integrates dx/dt = v_theta(t, x, o) from t=0 to t=1 with K steps.
Supports Euler and midpoint methods.
"""

import torch


class ODESolver:
    """
    Euler or midpoint ODE integration for flow matching inference.

    Integrates the learned velocity field from noise (t=0) to
    predicted actions (t=1).
    """

    def __init__(self, num_steps=50, method="euler"):
        self.num_steps = num_steps
        self.method = method

    def solve(self, velocity_fn, x0):
        """
        Integrate ODE from t=0 to t=1.

        Args:
            velocity_fn: callable(x_t, t) -> v, where
                x_t: (B, H, D) current state
                t: (B,) current time
                returns: (B, H, D) velocity
            x0: (B, H, D) initial noise at t=0

        Returns:
            x1: (B, H, D) predicted actions at t=1
        """
        dt = 1.0 / self.num_steps
        x = x0

        for i in range(self.num_steps):
            t = torch.full(
                (x.shape[0],), i * dt, device=x.device, dtype=x.dtype
            )

            if self.method == "euler":
                v = velocity_fn(x, t)
                x = x + dt * v

            elif self.method == "midpoint":
                v1 = velocity_fn(x, t)
                x_mid = x + 0.5 * dt * v1
                t_mid = t + 0.5 * dt
                v_mid = velocity_fn(x_mid, t_mid)
                x = x + dt * v_mid

            else:
                raise ValueError(f"Unknown ODE method: {self.method}")

        return x
