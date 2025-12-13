# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# """
# Channel layout (B, C, H, W):
# C0: type_A (soft prob)
# C1: type_B (soft prob)
# C2: adhesion in [0,1]
# C3: morphogen in [0,1]
# C4: center_mask in [0,1] (optional; passed through)

# Actions per-pixel (B, 3, H, W):
# A0: delta_adh ∈ [-0.1, 0.1]
# A1: v_x (small displacement)
# A2: v_y (small displacement)
# """

# TYPE_A = 0
# TYPE_B = 1
# ADH    = 2
# MORPH  = 3
# CENTER = 4


# class DCA(nn.Module):
#     def __init__(self, step_size=0.3, advect_scale=2.0, morphogen_diffusion=0.08):
#         super().__init__()
#         self.step_size = float(step_size)
#         self.advect_scale = float(advect_scale)
#         self.morphogen_diffusion = float(morphogen_diffusion)

#         # Perception over 3x3 neighborhood
#         self.percep = nn.Conv2d(5, 64, 3, padding=1, bias=True)

#         # Update network outputs residual deltas to the state (except center mask)
#         self.update = nn.Sequential(
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 1), nn.ReLU(),
#             nn.Conv2d(64, 4, 1)   # deltas for [type_A, type_B, adhesion, morphogen]
#         )

#         # Laplacian kernel for mild morphogen diffusion
#         lap = torch.tensor([[0.05, 0.2, 0.05],
#                             [0.2, -1.0, 0.2],
#                             [0.05, 0.2, 0.05]], dtype=torch.float32)
#         self.register_buffer("laplacian", lap.view(1, 1, 3, 3))

#         # 4-neighborhood mismatch kernel
#         k4 = torch.tensor([[0., 1., 0.],
#                            [1., 0., 1.],
#                            [0., 1., 0.]], dtype=torch.float32)
#         self.register_buffer("k4", k4.view(1, 1, 3, 3))

#     # ------------------------------------------------------------
#     # Normalization & helpers
#     # ------------------------------------------------------------
#     @staticmethod
#     def _soft_onehot_norm(types: torch.Tensor):
#         return F.softmax(types, dim=1)

#     def _morphogen_diffuse(self, m: torch.Tensor):
#         return m + self.morphogen_diffusion * F.conv2d(m, self.laplacian, padding=1)

#     # ------------------------------------------------------------
#     # Apply agent actions (adhesion update + advection)
#     # ------------------------------------------------------------
#     def _apply_actions(self, state: torch.Tensor, actions: torch.Tensor):
#         # state: (B, C, H, W)
#         # actions: expected (B, 3, H, W)
#         B, C, H, W = state.shape
#         # guard shapes
#         if actions.dim() == 3:  # maybe (B, N, A) or (B, A, N)
#             # try to interpret as (B, A, H*W) or (B, N, A)
#             if actions.shape[1] == 3 and actions.shape[2] == H * W:
#                 actions = actions.reshape(B, 3, H, W)
#             elif actions.shape[2] == 3 and actions.shape[1] == H * W:
#                 actions = actions.transpose(1, 2).reshape(B, 3, H, W)
#             else:
#                 raise ValueError(f"Cannot interpret actions with shape {actions.shape} in DCA._apply_actions")

#         if actions.dim() != 4 or actions.shape[1] != 3:
#             raise ValueError(f"Expect actions as (B,3,H,W) but got {actions.shape}")

#         delta_adh = actions[:, 0:1, :, :]
#         vx        = actions[:, 1:2, :, :]
#         vy        = actions[:, 2:3, :, :]

#         # ----- stronger adhesion update -----
#         adh_delta_scaled = (2.0 * delta_adh).clamp(-0.2, 0.2)  # action originally in [-0.1,0.1]
#         state[:, ADH:ADH+1, :, :] = (state[:, ADH:ADH+1, :, :] + adh_delta_scaled).clamp(0.0, 1.0)

#         # advection grid: amplify by advect_scale
#         vxs = self.advect_scale * vx
#         vys = self.advect_scale * vy

#         # build base grid (x,y) in [-1,1]
#         ys = torch.linspace(-1, 1, H, device=state.device, dtype=state.dtype)
#         xs = torch.linspace(-1, 1, W, device=state.device, dtype=state.dtype)
#         ys_grid, xs_grid = torch.meshgrid(ys, xs, indexing='ij')  # shape (H,W)
#         base_grid = torch.stack((xs_grid, ys_grid), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,H,W,2)

#         # normalize velocity to grid coordinates
#         vx_norm = vxs / max(W - 1, 1) * 2.0  # (B,1,H,W)
#         vy_norm = vys / max(H - 1, 1) * 2.0
#         flow = torch.stack((vx_norm.squeeze(1), vy_norm.squeeze(1)), dim=-1)  # (B,H,W,2)

#         new_grid = (base_grid + flow).clamp(-1.05, 1.05)

#         # warp channels except center mask
#         movable = state[:, :4, :, :]
#         warped = F.grid_sample(movable, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
#         state[:, :4, :, :] = warped
#         return state

#     # ------------------------------------------------------------
#     # Adhesion-based interfacial energy reduction
#     # ------------------------------------------------------------
#     def _adhesion_energy_delta(self, types: torch.Tensor, adhesion: torch.Tensor):
#         a = types[:, TYPE_A:TYPE_A + 1, :, :]
#         b = types[:, TYPE_B:TYPE_B + 1, :, :]
#         diff = a - b
#         nb = F.conv2d(diff, self.k4, padding=1)
#         press = adhesion * nb
#         delta_a = -press
#         delta_b = +press
#         return torch.cat([delta_a, delta_b], dim=1)

#     # ------------------------------------------------------------
#     # One DCA step
#     # ------------------------------------------------------------
#     def step(self, state: torch.Tensor, actions: torch.Tensor):
#         # 1) apply agent actions
#         state = self._apply_actions(state, actions)

#         # 2) learned local residual update
#         perception = self.percep(state)
#         residual = self.update(perception)
#         state[:, :4, :, :] = state[:, :4, :, :] + self.step_size * torch.tanh(residual)

#         # 3) adhesion-energy minimization (made stronger)
#         types = self._soft_onehot_norm(state[:, :2, :, :])
#         adhesion = state[:, ADH:ADH + 1, :, :]
#         d_types = self._adhesion_energy_delta(types, adhesion)

#         state[:, :2, :, :] = self._soft_onehot_norm(types + 0.4 * d_types)

#         # --------------------------------------------
#         # Adhesion-driven physical migration (amplified)
#         # --------------------------------------------
#         a = state[:, TYPE_A:TYPE_A + 1, :, :]
#         b = state[:, TYPE_B:TYPE_B + 1, :, :]
#         diff = a - b  # (B,1,H,W)

#         # Sobel filters (edge-based gradients)
#         sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=state.device).view(1, 1, 3, 3)
#         sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=state.device).view(1, 1, 3, 3)

#         gx = F.conv2d(diff, sobel_x, padding=1)
#         gy = F.conv2d(diff, sobel_y, padding=1)

#         adh = state[:, ADH:ADH + 1, :, :]

#         vx_adh = -adh * gx
#         vy_adh = -adh * gy

#         # normalize displacement (small)
#         scale = 0.08
#         vx_norm = vx_adh * scale
#         vy_norm = vy_adh * scale

#         B, _, H, W = state.shape
#         ys = torch.linspace(-1, 1, H, device=state.device, dtype=state.dtype)
#         xs = torch.linspace(-1, 1, W, device=state.device, dtype=state.dtype)
#         ys_grid, xs_grid = torch.meshgrid(ys, xs, indexing='ij')
#         base = torch.stack((xs_grid, ys_grid), dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  # (B,H,W,2)

#         vx_pix = vx_norm / max(W - 1, 1) * 2.0
#         vy_pix = vy_norm / max(H - 1, 1) * 2.0
#         flow = torch.stack((vx_pix.squeeze(1), vy_pix.squeeze(1)), dim=-1)  # (B,H,W,2)

#         new_grid = (base + flow).clamp(-1.05, 1.05)
#         types_warped = F.grid_sample(state[:, :2, :, :], new_grid, mode='bilinear', padding_mode='border', align_corners=True)
#         state[:, :2, :, :] = types_warped

#         # 4) morphogen: diffuse + soften dominance
#         m = state[:, MORPH:MORPH + 1, :, :]
#         m = self._morphogen_diffuse(m)
#         m = m.clamp(0.0, 1.0)
#         m = 0.8 * m
#         state[:, MORPH:MORPH + 1, :, :] = m

#         # keep adhesion bounded
#         state[:, ADH:ADH + 1, :, :] = state[:, ADH:ADH + 1, :, :].clamp(0.0, 1.0)

#         return state

#     def forward(self, state: torch.Tensor, actions: torch.Tensor, steps: int = 1):
#         s = state
#         for _ in range(int(steps)):
#             s = self.step(s, actions)
#         return s

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
    def __init__(self, step_size=0.3, advect_scale=2.0, morphogen_diffusion=0.08):
        """
        Increased advect_scale and step_size compared to some defaults so actions
        have a stronger, but still stable, effect.
        """
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
        """
        state: (B, C, H, W)
        actions: (B, 3, H, W)
        """
        B, C, H, W = state.shape
        # expect actions already scaled by env if desired
        delta_adh = actions[:, 0:1]
        vx        = actions[:, 1:2]
        vy        = actions[:, 2:3]

        # ----- stronger adhesion update -----
        # scale delta_adh so agent can change adhesion more aggressively (but still clamped)
        adh_delta_scaled = (2.0 * delta_adh).clamp(-0.2, 0.2)  # action originally in [-0.1,0.1]
        state[:, ADH:ADH+1] = (state[:, ADH:ADH+1] + adh_delta_scaled).clamp(0.0, 1.0)

        # advection grid: amplify by advect_scale
        vxs = self.advect_scale * vx
        vys = self.advect_scale * vy

        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=state.device),
            torch.linspace(-1, 1, W, device=state.device),
            indexing='ij'
        )
        base_grid = torch.stack((xs, ys), dim=-1).unsqueeze(0).repeat(B,1,1,1)

        # normalize velocity to [-2,2] range over the grid mapping
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
        # residual is for 4 channels: types (2), adhesion, morphogen
        state[:, :4] = state[:, :4] + self.step_size * torch.tanh(residual)

        # 3) adhesion-energy minimization
        types = self._soft_onehot_norm(state[:, :2])
        adhesion = state[:, ADH:ADH+1]
        d_types = self._adhesion_energy_delta(types, adhesion)

        # Increase coefficient so adhesion energy has visible effect on types
        state[:, :2] = self._soft_onehot_norm(types + 0.4 * d_types)

        # --------------------------------------------
        # Adhesion-driven physical migration (amplified)
        # --------------------------------------------
        a = state[:, TYPE_A:TYPE_A+1]
        b = state[:, TYPE_B:TYPE_B+1]
        diff = a - b  # positive = A-dominant, negative = B-dominant

        # Sobel filters (edge-based gradients)
        sobel_x = torch.tensor([[-1,0,1],
                                [-2,0,2],
                                [-1,0,1]], dtype=torch.float32, device=state.device).view(1,1,3,3)

        sobel_y = torch.tensor([[-1,-2,-1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], dtype=torch.float32, device=state.device).view(1,1,3,3)

        gx = F.conv2d(diff, sobel_x, padding=1)
        gy = F.conv2d(diff, sobel_y, padding=1)

        # adhesion-modulated velocity
        adh = state[:, ADH:ADH+1]
        vx_adh = -adh * gx
        vy_adh = -adh * gy

        # normalize displacement
        scale = 0.08  # migration scale; small but visible
        vx_norm = vx_adh * scale
        vy_norm = vy_adh * scale

        # build sampling grid
        B, _, H, W = state.shape
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=state.device),
            torch.linspace(-1, 1, W, device=state.device),
            indexing='ij'
        )
        base = torch.stack((xs, ys), dim=-1).unsqueeze(0).repeat(B,1,1,1)

        vx_pix = vx_norm / (W-1) * 2.0
        vy_pix = vy_norm / (H-1) * 2.0
        flow = torch.stack((vx_pix.squeeze(1), vy_pix.squeeze(1)), dim=-1)

        # Warp type channels only
        new_grid = (base + flow).clamp(-1.05, 1.05)
        types = state[:, :2]
        warped = F.grid_sample(types, new_grid, mode='bilinear', padding_mode='border', align_corners=True)
        state[:, :2] = warped

        # 4) morphogen: diffuse + soften dominance
        m = state[:, MORPH:MORPH+1]
        m = self._morphogen_diffuse(m)
        m = m.clamp(0.0, 1.0)

        # retain a softened morphogen but not too tiny
        m = 0.8 * m

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
