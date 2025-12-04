# # src/envs/wrappers.py
# import torch
# import torch.nn.functional as F
# from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
# from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches

# class SortingEnv:
#     """
#     Cell-sorting environment with both global and local (5x5x4) observation modes.

#     Modes:
#       - 'global': returns full grid state (B,5,H,W)
#       - 'local': returns per-cell patches (B,N,4,5,5) + coords
#     """

#     def __init__(self, H=64, W=64, device='cpu', gamma_motion=0.1,
#                  steps_per_action=1, obs_mode='local'):
#         self.H, self.W = H, W
#         self.device = torch.device(device)
#         self.gamma_motion = gamma_motion
#         self.steps_per_action = steps_per_action
#         self.obs_mode = obs_mode
#         self.dca = DCA().to(self.device)
#         self.state = None

#     # ----- state initialization -----
#     def _make_morphogen(self, B):
#         x = torch.linspace(0, 1, self.W, device=self.device).view(1,1,1,self.W).repeat(B,1,self.H,1)
#         return x

#     def reset(self, B=1, pA=0.5):
#         types = torch.rand(B, 2, self.H, self.W, device=self.device)
#         types = F.softmax(types, dim=1)
#         types[:, TYPE_A] = types[:, TYPE_A] * 0.5 + pA
#         types[:, TYPE_B] = types[:, TYPE_B] * 0.5 + (1 - pA)
#         types = F.softmax(types, dim=1)

#         adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
#         morphogen = self._make_morphogen(B)
#         center = torch.ones(B, 1, self.H, self.W, device=self.device)
#         # build state and immediately detach/clone to prevent autograd tracking
#         state = torch.cat([types, adhesion, morphogen, center], dim=1)
#         self.state = state.detach().clone()
#         return self.get_observation()

#     # ----- observation retrieval -----
#     def get_observation(self):
#         """
#         Depending on mode:
#           - global: returns full grid state (detached clone)
#           - local: returns (patches, coords) computed from detached clone
#         """
#         if self.obs_mode == 'global':
#             return self.state.detach().clone()
#         elif self.obs_mode == 'local':
#             # call extract_local_patches on a detached clone to be safe
#             patches, coords = extract_local_patches(self.state.detach().clone(), patch_size=5)
#             return patches, coords
#         else:
#             raise ValueError("obs_mode must be 'global' or 'local'")

#     # ----- stepping the environment -----
#     def step(self, actions):
#         """
#         actions:
#           - if global mode: tensor (B,3,H,W)
#           - if local mode: per-cell actions (B,N,3) that will be reshaped to grid
#         Returns:
#           observation, reward (detached), info (cpu tensors)
#         """
#         # defensive checks
#         if self.state is None:
#             raise RuntimeError("Environment not reset. Call reset() before step().")

#         B = self.state.shape[0]

#         if self.obs_mode == 'local':
#             # reshape per-cell actions to grid (B,3,H,W)
#             N = self.H * self.W
#             assert actions.shape[1] == N, f"expected actions second dim {N}, got {actions.shape[1]}"
#             # actions expected (B, N, A) -> transpose to (B, A, H, W)
#             actions = actions.transpose(1, 2).reshape(B, 3, self.H, self.W)

#         # Run environment dynamics under no_grad so env ops don't create graph
#         with torch.no_grad():
#             s = self.state
#             for _ in range(self.steps_per_action):
#                 # dca may modify or return new tensor; keep return and then detach/clone
#                 s = self.dca(s, actions, steps=1)
#             # ensure any underlying storage is detached and independent
#             self.state = s.detach().clone()

#             # compute metrics using detached tensors
#             e = interfacial_energy(self.state).detach()
#             mpen = motion_penalty(actions.detach()).detach()

#             reward = -(e + self.gamma_motion * mpen)
#             # info should be CPU tensors for logging
#             info = {
#                 "interfacial_energy": e.cpu(),
#                 "motion_penalty": mpen.cpu()
#             }

#         # return safe (detached) observation and reward
#         return self.get_observation(), reward.detach(), info

#     # ----- utility -----
#     def current_state(self):
#         """Return a detached copy of the full grid state (B,5,H,W)."""
#         return self.state.detach().clone()

# src/envs/wrappers.py
import torch
import torch.nn.functional as F
from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from ..utils.metrics import motion_penalty, extract_local_patches

class SortingEnv:
    """
    Cell-sorting environment with both global and local (5x5xC) observation modes.

    Modes:
      - 'global': returns full grid state (B,C,H,W)
      - 'local': returns (patches, coords) where patches is (B,N,C,5,5)
    """

    def __init__(
        self,
        H=64,
        W=64,
        device="cpu",
        gamma_motion=0.1,
        steps_per_action=1,
        obs_mode="local",
    ):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.gamma_motion = gamma_motion
        self.steps_per_action = steps_per_action
        self.obs_mode = obs_mode
        self.dca = DCA().to(self.device)
        self.state = None  # (B, C, H, W)

    # ----- state initialization -----
    def _make_morphogen(self, B):
        x = torch.linspace(0, 1, self.W, device=self.device)
        x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
        return x

    def reset(self, B=1, pA=0.5):
        """
        Initialize a random mixed state and return initial observation.
        """
        types = torch.rand(B, 2, self.H, self.W, device=self.device)
        types = F.softmax(types, dim=1)

        # bias toward pA for TYPE_A and (1-pA) for TYPE_B
        types[:, TYPE_A] = types[:, TYPE_A] * 0.5 + pA
        types[:, TYPE_B] = types[:, TYPE_B] * 0.5 + (1.0 - pA)
        types = F.softmax(types, dim=1)

        adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
        morphogen = self._make_morphogen(B)
        center = torch.ones(B, 1, self.H, self.W, device=self.device)

        # important: detach so env state is *not* part of any autograd graph
        self.state = torch.cat([types, adhesion, morphogen, center], dim=1).detach()

        return self.get_observation()

    # ----- observation retrieval -----
    def get_observation(self):
        """
        Depending on mode:
          - global: returns full grid state (B,C,H,W)
          - local: returns (patches, coords) where patches is (B,N,C,5,5)
        """
        if self.obs_mode == "global":
            # always return a detached clone
            return self.state.detach().clone()

        elif self.obs_mode == "local":
            # detach state before extracting patches -> no gradients through env
            state = self.state.detach()
            patches, coords = extract_local_patches(state, patch_size=5)
            return patches, coords

        else:
            raise ValueError("obs_mode must be 'global' or 'local'")

    # ----- sorting metric (for reward shaping) -----
    def _sorting_index(self, state):
        """
        Simple spatial sorting index for TYPE_A:
        |mean(TYPE_A on left half) - mean(TYPE_A on right half)|

        state: (B, C, H, W)
        returns: (B,) tensor, higher = stronger left/right segregation
        """
        A = state[:, TYPE_A]  # (B, H, W)
        mid = self.W // 2

        left = A[:, :, :mid].mean(dim=[1, 2])
        right = A[:, :, mid:].mean(dim=[1, 2])

        return torch.abs(left - right)

    # ----- stepping the environment -----
    def step(self, actions):
        """
        actions:
          - if global mode: tensor (B,3,H,W)
          - if local mode: per-cell actions (B,N,3) that will be reshaped to grid
        """
        B = self.state.shape[0]

        if self.obs_mode == "local":
            # reshape per-cell actions to grid (B,3,H,W)
            N = self.H * self.W
            assert actions.shape[1] == N, f"expected actions shape (B,N,3) with N={N}, got {actions.shape}"
            actions = actions.transpose(1, 2).reshape(B, 3, self.H, self.W)

        # ---- run DCA dynamics with no grad ----
        with torch.no_grad():
            s = self.state
            for _ in range(self.steps_per_action):
                s = self.dca(s, actions, steps=1)

        # detach so state is permanently kept outside any autograd graph
        self.state = s.detach()

        # ---- compute metrics (no grad needed) ----
        with torch.no_grad():
            # you removed interfacial_energy import; keep reward from motion + sorting
            mpen = motion_penalty(actions)  # (B,)
            sorting_index = self._sorting_index(self.state)  # (B,)

            # reward = +sorting - motion_penalty * gamma
            reward = sorting_index - self.gamma_motion * mpen

        info = {
            "motion_penalty": mpen.detach().cpu(),
            "sorting_index": sorting_index.detach().cpu(),
        }

        # return next observation and *detached* reward
        return self.get_observation(), reward.detach(), info

    # ----- utility -----
    def current_state(self):
        """Return a detached copy of the full grid state (B,C,H,W)."""
        return self.state.detach().clone()
