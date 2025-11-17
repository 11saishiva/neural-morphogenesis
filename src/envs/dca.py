import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Channel layout (B, C, H, W):
C0: type_A (soft prob)
C1: type_B (soft prob)
C2: adhesion in [0,1]
C3: morphogen in [0,1]
C4: center_mask in [0,1] (optional; passed through)

Actions per-pixel (B, 3, H, W):
A0: delta_adh ∈ [-0.1, 0.1]
A1: v_x (small displacement)
A2: v_y (small displacement)
"""

TYPE_A = 0
TYPE_B = 1
ADH    = 2
MORPH  = 3
CENTER = 4


class DCA(nn.Module):
    def __init__(self, step_size=0.2, advect_scale=0.5, morphogen_diffusion=0.05):
        super().__init__()
        self.step_size = step_size
        self.advect_scale = advect_scale
        self.morphogen_diffusion = morphogen_diffusion

        # Perception over 3x3 neighborhood
        self.percep = nn.Conv2d(5, 64, 3, padding=1, bias=True)

        # Update network outputs residual deltas to the state (except center mask)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 4, 1)   # deltas for [type_A, type_B, adhesion, morphogen]
        )

        # Laplacian kernel for mild morphogen diffusion
        lap = torch.tensor([[0.05, 0.2, 0.05],
                            [0.2, -1.0, 0.2],
                            [0.05, 0.2, 0.05]], dtype=torch.float32)
        self.register_buffer("laplacian", lap.view(1,1,3,3))

        # 4-neighborhood mismatch kernel
        k4 = torch.tensor([[0., 1., 0.],
                           [1., 0., 1.],
                           [0., 1., 0.]], dtype=torch.float32)
        self.register_buffer("k4", k4.view(1,1,3,3))

    # ------------------------------------------------------------
    # Normalization & helpers
    # ------------------------------------------------------------
    @staticmethod
    def _soft_onehot_norm(types):
        return F.softmax(types, dim=1)

    def _morphogen_diffuse(self, m):
        return m + self.morphogen_diffusion * F.conv2d(m, self.laplacian, padding=1)

    # ------------------------------------------------------------
    # Apply agent actions (adhesion update + advection)
    # ------------------------------------------------------------
    def _apply_actions(self, state, actions):
        B, C, H, W = state.shape
        delta_adh = actions[:, 0:1]
        vx        = actions[:, 1:2]
        vy        = actions[:, 2:3]

        # adhesion update
        state[:, ADH:ADH+1] = (state[:, ADH:ADH+1] + delta_adh.clamp(-0.1, 0.1)).clamp(0.0, 1.0)

        # advection grid
        vxs = self.advect_scale * vx
        vys = self.advect_scale * vy

        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=state.device),
            torch.linspace(-1, 1, W, device=state.device),
            indexing='ij'
        )
        base_grid = torch.stack((xs, ys), dim=-1).unsqueeze(0).repeat(B,1,1,1)

        vx_norm = vxs / max(W-1, 1) * 2.0
        vy_norm = vys / max(H-1, 1) * 2.0
        flow = torch.stack((vx_norm.squeeze(1), vy_norm.squeeze(1)), dim=-1)

        new_grid = (base_grid + flow).clamp(-1.05, 1.05)

        # warp channels except center mask
        movable = state[:, :4]
        warped = F.grid_sample(movable, new_grid, mode='bilinear',
                               padding_mode='border', align_corners=True)
        state[:, :4] = warped
        return state

    # ------------------------------------------------------------
    # Adhesion-based interfacial energy reduction
    # ------------------------------------------------------------
    def _adhesion_energy_delta(self, types, adhesion):
        a = types[:, TYPE_A:TYPE_A+1]
        b = types[:, TYPE_B:TYPE_B+1]
        diff = a - b

        nb = F.conv2d(diff, self.k4, padding=1)
        press = adhesion * nb
        delta_a = -press
        delta_b = +press
        return torch.cat([delta_a, delta_b], dim=1)

    # ------------------------------------------------------------
    # One DCA step
    # ------------------------------------------------------------
    def step(self, state, actions):

        # 1) apply agent actions
        state = self._apply_actions(state, actions)

        # 2) learned local residual update
        perception = self.percep(state)
        residual   = self.update(perception)
        state[:, :4] = state[:, :4] + self.step_size * torch.tanh(residual)

        # 3) adhesion-energy minimization
        types = self._soft_onehot_norm(state[:, :2])
        adhesion = state[:, ADH:ADH+1]
        d_types = self._adhesion_energy_delta(types, adhesion)
        state[:, :2] = self._soft_onehot_norm(types + 0.1 * d_types)

        # 4) morphogen: diffuse + SOFTEN DOMINANCE
        m = state[:, MORPH:MORPH+1]
        m = self._morphogen_diffuse(m)
        m = m.clamp(0.0, 1.0)

        # ★★★ APPLY MORPHOGEN SOFTENING HERE ★★★
        # reduces strength without breaking gradient flow
        m = 0.5 * m

        state[:, MORPH:MORPH+1] = m

        # keep adhesion bounded
        state[:, ADH:ADH+1] = state[:, ADH:ADH+1].clamp(0.0, 1.0)

        return state

    # ------------------------------------------------------------
    def forward(self, state, actions, steps=1):
        s = state
        for _ in range(steps):
            s = self.step(s, actions)
        return s
