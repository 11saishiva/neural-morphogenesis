# src/envs/wrappers.py
import torch
import torch.nn.functional as F
from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches

class SortingEnv:
    """
    Cell-sorting environment with both global and local (5x5x4) observation modes.

    Modes:
      - 'global': returns full grid state (B,5,H,W)
      - 'local': returns per-cell patches (B,N,4,5,5) + coords
    """

    def __init__(self, H=64, W=64, device='cpu', gamma_motion=0.1,
                 steps_per_action=1, obs_mode='local'):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.gamma_motion = gamma_motion
        self.steps_per_action = steps_per_action
        self.obs_mode = obs_mode
        self.dca = DCA().to(self.device)
        self.state = None

    # ----- state initialization -----
    def _make_morphogen(self, B):
        x = torch.linspace(0, 1, self.W, device=self.device).view(1,1,1,self.W).repeat(B,1,self.H,1)
        return x

    def reset(self, B=1, pA=0.5):
        types = torch.rand(B, 2, self.H, self.W, device=self.device)
        types = F.softmax(types, dim=1)
        types[:, TYPE_A] = types[:, TYPE_A] * 0.5 + pA
        types[:, TYPE_B] = types[:, TYPE_B] * 0.5 + (1 - pA)
        types = F.softmax(types, dim=1)

        adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
        morphogen = self._make_morphogen(B)
        center = torch.ones(B, 1, self.H, self.W, device=self.device)
        self.state = torch.cat([types, adhesion, morphogen, center], dim=1)
        return self.get_observation()

    # ----- observation retrieval -----
    def get_observation(self):
        """
        Depending on mode:
          - global: returns full grid state
          - local: returns (patches, coords)
        """
        if self.obs_mode == 'global':
            return self.state.clone()
        elif self.obs_mode == 'local':
            patches, coords = extract_local_patches(self.state, patch_size=5)
            return patches, coords
        else:
            raise ValueError("obs_mode must be 'global' or 'local'")

    # ----- stepping the environment -----
    def step(self, actions):
        """
        actions:
          - if global mode: tensor (B,3,H,W)
          - if local mode: per-cell actions (B,N,3) that will be reshaped to grid
        """
        B = self.state.shape[0]

        if self.obs_mode == 'local':
            # reshape per-cell actions to grid (B,3,H,W)
            N = self.H * self.W
            assert actions.shape[1] == N
            actions = actions.transpose(1, 2).reshape(B, 3, self.H, self.W)

        s = self.state
        for _ in range(self.steps_per_action):
            s = self.dca(s, actions, steps=1)
        self.state = s

        e = interfacial_energy(self.state)
        mpen = motion_penalty(actions)
        reward = -e - self.gamma_motion * mpen
        info = {
            "interfacial_energy": e.detach().cpu(),
            "motion_penalty": mpen.detach().cpu()
        }
        return self.get_observation(), reward.detach(), info
        # ----- utility -----
    def current_state(self):
        """Return a detached copy of the full grid state (B,5,H,W)."""
        return self.state.clone().detach()


        return self.state.clone()
