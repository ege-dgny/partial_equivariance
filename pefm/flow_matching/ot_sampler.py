"""
Optimal Transport conditional flow matching.

Implements linear interpolation paths (OT conditional):
    x_t = (1-t) * x_0 + t * x_1
    u_t = x_1 - x_0  (constant velocity along OT path)

No noise scheduler needed -- just linear interpolation.
"""

import torch


class OTConditionalFlowMatching:
    """
    OT Conditional Flow Matching for training flow-based policies.

    Given noise x_0 and target x_1, defines:
    - Path: x_t = (1-t)*x_0 + t*x_1
    - Target velocity: u_t = x_1 - x_0

    Optional: add small Gaussian noise sigma_min to the interpolant
    for numerical stability.
    """

    def __init__(self, sigma_min=0.001):
        self.sigma_min = sigma_min

    def sample_xt(self, x0, x1, t):
        """
        Sample point along OT path.

        x0: (B, H, D) noise
        x1: (B, H, D) target action
        t:  (B, 1, 1) time in [0, 1]
        returns: x_t = (1-t)*x0 + t*x1 + sigma_min * noise
        """
        x_t = (1 - t) * x0 + t * x1
        if self.sigma_min > 0:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)
        return x_t

    def target_velocity(self, x0, x1):
        """
        Target velocity field (constant along OT path).

        Returns u_t = x1 - x0
        """
        return x1 - x0
