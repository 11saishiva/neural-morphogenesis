# import torch
# import torch.nn.functional as F
# from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
# from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches


# class SortingEnv:
#     """
#     Stable Sorting environment.
#     Obs modes:
#       - 'local'  -> get_observation() returns (patches, coords)
#       - 'global' -> get_observation() returns full state (B,C,H,W)

#     Actions expected:
#       - local: (B, N, A) where N = H*W, A = 3 (delta_adh, vx, vy)
#       - global: (B, 3, H, W)
#     """

#     def __init__(
#         self, H=64, W=64, device='cpu', gamma_motion=0.1,
#         steps_per_action=6, obs_mode='local'
#     ):
#         self.H, self.W = int(H), int(W)
#         self.device = torch.device(device)
#         self.gamma_motion = float(gamma_motion)
#         self.steps_per_action = int(steps_per_action)
#         assert obs_mode in ('local', 'global')
#         self.obs_mode = obs_mode

#         # Dynamics model
#         self.dca = DCA().to(self.device)
#         self.state = None

#         # Reward shaping
#         self.sort_weight = 150.0
#         self.sort_bonus = 2.0
#         self.energy_weight = 1.0
#         self.motion_weight = 0.03

#         self.term_clip = 20.0
#         self.reward_clip = 20.0

#         self.sort_ema_alpha = 0.20
#         self._sort_ema = None
#         self._last_sort_idx = None

#         self.pos_delta_rms_alpha = 0.05
#         self._pos_delta_rms = None
#         self._pos_delta_eps = 1e-6

#         self.SORT_AMPLIFY = 100.0
#         self._running_scale_floor = 1e-4

#         self._norm_pos_delta_ema = None
#         self._norm_pos_delta_ema_alpha = 0.5

#         # diagnostics
#         self._raw_sort_ma20 = None
#         self._raw_sort_ma50 = None
#         self._env_step = 0

#     def _make_morphogen(self, B):
#         x = torch.linspace(0, 1, self.W, device=self.device, dtype=torch.float32)
#         x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
#         return x

#     def reset(self, B=1, pA=0.5):
#         B = int(B)
#         pA = float(pA)

#         # initialize soft type probabilities biased by pA
#         types = torch.rand(B, 2, self.H, self.W, device=self.device)
#         types = F.softmax(types, dim=1)
#         types[:, TYPE_A, :, :] = types[:, TYPE_A, :, :] * 0.5 + pA
#         types[:, TYPE_B, :, :] = types[:, TYPE_B, :, :] * 0.5 + (1.0 - pA)
#         types = F.softmax(types, dim=1)

#         adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
#         morphogen = self._make_morphogen(B)
#         center = torch.ones(B, 1, self.H, self.W, device=self.device)

#         state = torch.cat([types, adhesion, morphogen, center], dim=1)
#         self.state = state.detach().clone()

#         # trackers
#         self._sort_ema = torch.zeros(B, device=self.device)
#         self._pos_delta_rms = torch.ones(B, device=self.device)
#         self._norm_pos_delta_ema = torch.zeros(B, device=self.device)

#         with torch.no_grad():
#             raw = self._sorting_index(self.state).detach()
#             self._last_sort_idx = (raw * self.SORT_AMPLIFY).clone()
#             self._raw_sort_ma20 = raw.clone()
#             self._raw_sort_ma50 = raw.clone()

#         self._env_step = 0
#         return self.get_observation()

#     def get_observation(self):
#         if self.obs_mode == 'global':
#             return self.state.detach().clone()
#         else:
#             patches, coords = extract_local_patches(self.state.detach().clone(), patch_size=5)
#             return patches, coords

#     def _sorting_index(self, state):
#         # state: (B, C, H, W)
#         A = state[:, TYPE_A, :, :]  # (B,H,W)
#         mid = self.W // 2
#         left = A[:, :, :mid].mean(dim=[1, 2]) if mid > 0 else torch.zeros(A.shape[0], device=A.device)
#         right = A[:, :, mid:].mean(dim=[1, 2]) if self.W - mid > 0 else torch.zeros(A.shape[0], device=A.device)
#         return torch.abs(left - right)

#     def step(self, actions):
#         if self.state is None:
#             raise RuntimeError("Call reset() before step().")

#         B = self.state.shape[0]
#         self._env_step += 1

#         # Validate & reshape actions into (B,3,H,W)
#         if self.obs_mode == 'local':
#             if not (isinstance(actions, torch.Tensor) and actions.dim() == 3 and actions.shape[0] == B):
#                 raise ValueError(f"For local mode expected actions (B, N, A); got {getattr(actions, 'shape', None)}")
#             N = self.H * self.W
#             if actions.shape[1] != N:
#                 raise ValueError(f"Local actions second dim must be H*W={N}; got {actions.shape[1]}")
#             # actions: (B, N, A) -> (B, A, N)
#             actions = actions.transpose(1, 2).reshape(B, 3, self.H, self.W)
#         else:
#             if not (isinstance(actions, torch.Tensor) and actions.dim() == 4 and actions.shape == (B, 3, self.H, self.W)):
#                 raise ValueError(f"For global mode expected (B,3,H,W). Got {getattr(actions, 'shape', None)}")

#         # ensure actions on same device/dtype
#         actions = actions.to(self.state.device).float()

#         with torch.no_grad():
#             # Apply dynamics (DCA) for steps_per_action iterations
#             s = self.state
#             for _ in range(self.steps_per_action):
#                 s = self.dca(s, actions, steps=1)
#             self.state = s.detach().clone()

#             # energy and motion
#             e = interfacial_energy(self.state)           # (B,)
#             mpen = motion_penalty(actions.detach())      # (B,)

#             # sorting index and amplify
#             raw_sort_idx = self._sorting_index(self.state)  # (B,)
#             sort_idx = raw_sort_idx * self.SORT_AMPLIFY

#             if self._last_sort_idx is None:
#                 delta_sort = torch.zeros_like(sort_idx)
#             else:
#                 delta_sort = sort_idx - self._last_sort_idx

#             self._last_sort_idx = sort_idx.detach().clone()

#             # EMA of delta-sort
#             if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
#                 self._sort_ema = torch.zeros_like(sort_idx)

#             alpha = float(self.sort_ema_alpha)
#             self._sort_ema = (1 - alpha) * self._sort_ema.to(sort_idx.device) + alpha * delta_sort

#             pos_delta = torch.relu(self._sort_ema)  # (B,)

#             # RMS normalization of pos_delta (running variance-like)
#             if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
#                 self._pos_delta_rms = torch.ones_like(pos_delta)

#             beta = float(self.pos_delta_rms_alpha)
#             sq = pos_delta ** 2
#             self._pos_delta_rms = (1 - beta) * self._pos_delta_rms.to(pos_delta.device) + beta * sq
#             running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

#             # enforce a minimum floor (prevents huge norm when running_scale tiny)
#             running_scale = torch.max(running_scale, torch.full_like(running_scale, self._running_scale_floor))

#             norm_pos_delta = pos_delta / (running_scale + self._pos_delta_eps)  # (B,)

#             # small EMA on norm_pos_delta so it doesn't instantly become exactly zero
#             if self._norm_pos_delta_ema is None or self._norm_pos_delta_ema.shape[0] != norm_pos_delta.shape[0]:
#                 self._norm_pos_delta_ema = torch.zeros_like(norm_pos_delta)
#             nna = float(self._norm_pos_delta_ema_alpha)
#             self._norm_pos_delta_ema = (1 - nna) * self._norm_pos_delta_ema.to(norm_pos_delta.device) + nna * norm_pos_delta
#             stable_norm_pos_delta = self._norm_pos_delta_ema

#             # moving averages (diagnostics)
#             if (self._raw_sort_ma20 is None) or (self._raw_sort_ma20.shape[0] != raw_sort_idx.shape[0]):
#                 self._raw_sort_ma20 = raw_sort_idx.clone()
#                 self._raw_sort_ma50 = raw_sort_idx.clone()
#             else:
#                 alpha20 = 2.0 / (20.0 + 1.0)
#                 alpha50 = 2.0 / (50.0 + 1.0)
#                 self._raw_sort_ma20 = (1.0 - alpha20) * self._raw_sort_ma20 + alpha20 * raw_sort_idx
#                 self._raw_sort_ma50 = (1.0 - alpha50) * self._raw_sort_ma50 + alpha50 * raw_sort_idx

#             # reward composition
#             sort_term = self.sort_weight * stable_norm_pos_delta
#             bonus_term = self.sort_bonus * raw_sort_idx
#             energy_term = -self.energy_weight * e
#             motion_term = -self.motion_weight * mpen

#             # clip per-term
#             sort_term = torch.clamp(sort_term, -self.term_clip, self.term_clip)
#             bonus_term = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
#             energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
#             motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

#             reward = sort_term + bonus_term + energy_term + motion_term
#             reward = reward / float(self.steps_per_action)
#             reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

#             # helper to get scalars for logging
#             def _scalar(x):
#                 if isinstance(x, torch.Tensor):
#                     return float(x.detach().cpu().mean())
#                 try:
#                     return float(x)
#                 except Exception:
#                     return float(torch.as_tensor(x).cpu().mean())

#             # scalar diagnostics
#             raw_mean = _scalar(raw_sort_idx)
#             ma20_mean = _scalar(self._raw_sort_ma20)
#             ma50_mean = _scalar(self._raw_sort_ma50)
#             reward_mean = _scalar(reward)
#             norm_pos_delta_mean = _scalar(stable_norm_pos_delta)

#             if (self._env_step % 10) == 0:
#                 print(
#                     f"[SORT-MA] step={self._env_step} raw={raw_mean:.6e} "
#                     f"ma20={ma20_mean:.6e} ma50={ma50_mean:.6e} reward_mean={reward_mean:.6f} "
#                     f"norm_pos_delta={norm_pos_delta_mean:.6e}",
#                     flush=True,
#                 )

#             reward_components = {
#                 "reward_mean": reward_mean,
#                 "sort_term_mean": _scalar(sort_term),
#                 "bonus_term_mean": _scalar(bonus_term),
#                 "energy_term_mean": _scalar(energy_term),
#                 "motion_term_mean": _scalar(motion_term),
#                 "raw_sort_idx_mean": raw_mean,
#                 "sort_idx_mean": _scalar(sort_idx),
#                 "pos_delta_mean": _scalar(pos_delta),
#                 "norm_pos_delta_mean": norm_pos_delta_mean,
#                 "interfacial_energy_mean": _scalar(e),
#                 "motion_penalty_mean": _scalar(mpen),
#                 "running_scale_mean": _scalar(running_scale),
#                 "raw_sort_ma20_mean": ma20_mean,
#                 "raw_sort_ma50_mean": ma50_mean,
#                 "steps_per_action": float(self.steps_per_action),
#             }

#             info = {
#                 "interfacial_energy": e.cpu(),
#                 "motion_penalty": mpen.cpu(),
#                 "raw_sort_index": raw_sort_idx.cpu(),
#                 "sort_index": sort_idx.cpu(),
#                 "delta_sort_index": delta_sort.cpu(),
#                 "smoothed_delta": self._sort_ema.cpu(),
#                 "pos_delta": pos_delta.cpu(),
#                 "pos_delta_rms": self._pos_delta_rms.cpu(),
#                 "running_scale": running_scale.cpu(),
#                 "norm_pos_delta": stable_norm_pos_delta.cpu(),
#                 "sort_term": sort_term.cpu(),
#                 "bonus_term": bonus_term.cpu(),
#                 "energy_term": energy_term.cpu(),
#                 "motion_term": motion_term.cpu(),
#                 "raw_sort_ma20": self._raw_sort_ma20.cpu(),
#                 "raw_sort_ma50": self._raw_sort_ma50.cpu(),
#                 "reward_components": reward_components,
#             }

#         return self.get_observation(), reward.detach(), info

#     def current_state(self):
#         return self.state.detach().clone()

import torch
import torch.nn.functional as F
from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches


class SortingEnv:
    """
    Stable, production-ready Sorting environment.

    Observation modes:
      - 'local'  -> get_observation() returns (patches, coords)
      - 'global' -> get_observation() returns full state (B,C,H,W)

    Actions expected:
      - local: (B, N, A) where N = H*W, A = 3 (delta_adh, vx, vy)
      - global: (B, 3, H, W)
    """

    def __init__(
        self, H=64, W=64, device='cpu', gamma_motion=0.1,
        steps_per_action=6, obs_mode='local'
    ):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.gamma_motion = gamma_motion
        self.steps_per_action = steps_per_action
        assert obs_mode in ('local', 'global')
        self.obs_mode = obs_mode

        # Dynamics model
        self.dca = DCA().to(self.device)
        self.state = None

        # --- Reward shaping (tuned for stability) ---
        # Stronger incentive for sorting, gentler motion cost to allow exploration
        self.sort_weight   = 300.0    # main multiplier for normalized sorting signal
        self.sort_bonus    = 3.0
        self.energy_weight = 1.0
        self.motion_weight = 0.009    # reduced motion penalty to not drown small sort gains

        # clipping thresholds
        self.term_clip     = 20.0
        self.reward_clip   = 20.0

        # EMA smoothing of delta_sort
        self.sort_ema_alpha = 0.20
        self._sort_ema = None
        self._last_sort_idx = None

        # RMS normalizer for pos_delta (running variance-like)
        self.pos_delta_rms_alpha = 0.12
        self._pos_delta_rms = None
        self._pos_delta_eps = 1e-6

        # Critical scaling factor for sorting index
        self.SORT_AMPLIFY = 80.0

        # Small lower floor for running_scale to avoid divide-by-tiny
        self._running_scale_floor = 1e-6

        # Small EMA for norm_pos_delta so it can't vanish instantly
        self._norm_pos_delta_ema = None
        self._norm_pos_delta_ema_alpha = 0.3

        # Environment-side action scaling (train-time boost to make actions more effective)
        self._action_env_scale = 1.8

        # Diagnostics
        self._raw_sort_ma20 = None
        self._raw_sort_ma50 = None
        self._env_step = 0

    # -------------------------
    # Helpers
    # -------------------------
    def _make_morphogen(self, B):
        x = torch.linspace(0, 1, self.W, device=self.device)
        x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
        return x

    def reset(self, B=1, pA=0.5):
        # initialize types with soft probabilities biased by pA, with small per-sample jitter
        pA = float(pA)
        # jitter pA slightly per sample so some examples have exploitable asymmetry
        pA_jitter = (torch.rand(B, device=self.device) - 0.5) * 0.08  # +/-0.04 jitter
        pA_vec = (pA + pA_jitter).clamp(0.05, 0.95)  # keep sensible bounds

        types = torch.rand(B, 2, self.H, self.W, device=self.device)
        types = F.softmax(types, dim=1)
        # apply per-sample bias using pA_vec
        types[:, TYPE_A, :, :] = types[:, TYPE_A, :, :] * 0.5 + pA_vec.view(B, 1, 1).expand(-1, self.H, self.W)
        types[:, TYPE_B, :, :] = types[:, TYPE_B, :, :] * 0.5 + (1.0 - pA_vec).view(B, 1, 1).expand(-1, self.H, self.W)
        types = F.softmax(types, dim=1)

        adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
        morphogen = self._make_morphogen(B)
        center = torch.ones(B, 1, self.H, self.W, device=self.device)

        state = torch.cat([types, adhesion, morphogen, center], dim=1)
        self.state = state.detach().clone()

        B_actual = state.shape[0]

        # initialize EMA and RMS trackers
        self._sort_ema = torch.zeros(B_actual, device=self.device)
        self._pos_delta_rms = torch.ones(B_actual, device=self.device) * 1.0
        self._norm_pos_delta_ema = torch.zeros(B_actual, device=self.device)

        # initial sorting index baseline
        with torch.no_grad():
            raw = self._sorting_index(self.state).detach()
            self._last_sort_idx = (raw * self.SORT_AMPLIFY).clone()
            # initialize moving averages for diagnostics
            self._raw_sort_ma20 = raw.clone()
            self._raw_sort_ma50 = raw.clone()

        self._env_step = 0
        return self.get_observation()

    def get_observation(self):
        if self.obs_mode == 'global':
            return self.state.detach().clone()
        else:
            patches, coords = extract_local_patches(
                self.state.detach().clone(), patch_size=5
            )
            return patches, coords

    def _sorting_index(self, state):
        # state: (B, C, H, W)
        A = state[:, TYPE_A]  # (B, H, W)
        mid = self.W // 2
        left = A[:, :, :mid].mean(dim=[1, 2])
        right = A[:, :, mid:].mean(dim=[1, 2])
        return torch.abs(left - right)

    # -------------------------
    # Step
    # -------------------------
    def step(self, actions):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        B = self.state.shape[0]
        self._env_step += 1

        # Validate & reshape actions
        if self.obs_mode == 'local':
            if not (actions.dim() == 3 and actions.shape[0] == B):
                raise ValueError(f"For local mode expected actions (B, N, A); got {actions.shape}")
            N = self.H * self.W
            if actions.shape[1] != N:
                raise ValueError(f"Local actions second dim must be H*W={N}; got {actions.shape[1]}")
            # actions originally (B, N, A) -> (B, A, H, W)
            actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)
        else:
            if not (actions.dim() == 4 and actions.shape == (B, 3, self.H, self.W)):
                raise ValueError(f"For global mode expected (B,3,H,W). Got {actions.shape}")

        # scale actions slightly inside environment to make their effect stronger early on
        actions = (actions * float(self._action_env_scale)).to(self.state.device).float()

        with torch.no_grad():
            # Apply dynamics (DCA) for k steps
            s = self.state
            for _ in range(self.steps_per_action):
                s = self.dca(s, actions, steps=1)
            self.state = s.detach().clone()

            # energy and motion
            e = interfacial_energy(self.state).detach()           # (B,)
            mpen = motion_penalty(actions.detach()).detach()      # (B,)

            # sorting index and amplify
            raw_sort_idx = self._sorting_index(self.state).detach()  # (B,)
            sort_idx = raw_sort_idx * self.SORT_AMPLIFY

            if self._last_sort_idx is None:
                delta_sort = torch.zeros_like(sort_idx)
            else:
                delta_sort = sort_idx - self._last_sort_idx

            self._last_sort_idx = sort_idx.detach().clone()

            # debug printing infrequently to check delta_sort
            if (self._env_step % 50) == 0:
                ds = delta_sort.detach().cpu()
                try:
                    dmin = float(ds.min().item())
                    dmax = float(ds.max().item())
                except Exception:
                    dmin, dmax = 0.0, 0.0
                print(f"[DEBUG] env_step={self._env_step} delta_sort min/max = {dmin:.3e}/{dmax:.3e}", flush=True)

            # EMA of delta-sort
            if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
                self._sort_ema = torch.zeros_like(sort_idx)

            alpha = float(self.sort_ema_alpha)
            self._sort_ema = (1 - alpha) * self._sort_ema.to(sort_idx.device) + alpha * delta_sort

            pos_delta = torch.relu(self._sort_ema)  # (B,)

            # RMS normalization of pos_delta (running variance-like)
            if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
                self._pos_delta_rms = torch.ones_like(pos_delta)

            beta = float(self.pos_delta_rms_alpha)
            sq = pos_delta ** 2
            self._pos_delta_rms = (1 - beta) * self._pos_delta_rms.to(pos_delta.device) + beta * sq
            running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

            # enforce a sensible minimum floor (prevents huge norm when running_scale tiny)
            running_scale = torch.max(running_scale, torch.full_like(running_scale, self._running_scale_floor))

            norm_pos_delta = pos_delta / (running_scale + self._pos_delta_eps)  # (B,)

            # small EMA on norm_pos_delta so it doesn't instantly become exactly zero
            if self._norm_pos_delta_ema is None or self._norm_pos_delta_ema.shape[0] != norm_pos_delta.shape[0]:
                self._norm_pos_delta_ema = torch.zeros_like(norm_pos_delta)
            nna = float(self._norm_pos_delta_ema_alpha)
            self._norm_pos_delta_ema = (1 - nna) * self._norm_pos_delta_ema.to(norm_pos_delta.device) + nna * norm_pos_delta
            # use the EMA value as the normalized sort signal (more stable)
            stable_norm_pos_delta = self._norm_pos_delta_ema

            # -----------------------------
            # moving averages (diagnostics)
            # -----------------------------
            if (self._raw_sort_ma20 is None) or (self._raw_sort_ma20.shape[0] != raw_sort_idx.shape[0]):
                self._raw_sort_ma20 = raw_sort_idx.clone()
                self._raw_sort_ma50 = raw_sort_idx.clone()
            else:
                alpha20 = 2.0 / (20.0 + 1.0)
                alpha50 = 2.0 / (50.0 + 1.0)
                self._raw_sort_ma20 = (1.0 - alpha20) * self._raw_sort_ma20 + alpha20 * raw_sort_idx
                self._raw_sort_ma50 = (1.0 - alpha50) * self._raw_sort_ma50 + alpha50 * raw_sort_idx

            # -----------------------------
            # reward composition
            # -----------------------------
            # use stable_norm_pos_delta so we avoid spuriously zero sort_term
            sort_term   = self.sort_weight * stable_norm_pos_delta
            bonus_term  = self.sort_bonus * raw_sort_idx
            energy_term = -self.energy_weight * e
            motion_term = -self.motion_weight * mpen

            # clip per-term (element-wise)
            sort_term   = torch.clamp(sort_term, -self.term_clip, self.term_clip)
            bonus_term  = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
            energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
            motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

            reward = sort_term + bonus_term + energy_term + motion_term
            reward = reward / float(self.steps_per_action)
            reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

            # -----------------------------
            # scalar helpers that accept both tensors and floats
            # -----------------------------
            def _scalar(x):
                if isinstance(x, torch.Tensor):
                    return float(x.detach().cpu().mean())
                try:
                    return float(x)
                except Exception:
                    return float(torch.as_tensor(x).cpu().mean())

            # compute scalars (always produce python floats)
            raw_mean = _scalar(raw_sort_idx)
            ma20_mean = _scalar(self._raw_sort_ma20)
            ma50_mean = _scalar(self._raw_sort_ma50)
            reward_mean = _scalar(reward)
            sort_term_mean = _scalar(sort_term)
            bonus_term_mean = _scalar(bonus_term)
            energy_term_mean = _scalar(energy_term)
            motion_term_mean = _scalar(motion_term)
            pos_delta_mean = _scalar(pos_delta)
            norm_pos_delta_mean = _scalar(stable_norm_pos_delta)
            running_scale_mean = _scalar(running_scale)
            interfacial_energy_mean = _scalar(e)
            motion_penalty_mean = _scalar(mpen)

            # periodic minimal logging
            if (self._env_step % 10) == 0:
                print(
                    f"[SORT-MA] step={self._env_step} raw={raw_mean:.6e} "
                    f"ma20={ma20_mean:.6e} ma50={ma50_mean:.6e} reward_mean={reward_mean:.6f} "
                    f"norm_pos_delta={norm_pos_delta_mean:.6e}",
                    flush=True,
                )

            reward_components = {
                "reward_mean": reward_mean,
                "sort_term_mean": sort_term_mean,
                "bonus_term_mean": bonus_term_mean,
                "energy_term_mean": energy_term_mean,
                "motion_term_mean": motion_term_mean,
                "raw_sort_idx_mean": raw_mean,
                "sort_idx_mean": _scalar(sort_idx),
                "pos_delta_mean": pos_delta_mean,
                "norm_pos_delta_mean": norm_pos_delta_mean,
                "interfacial_energy_mean": interfacial_energy_mean,
                "motion_penalty_mean": motion_penalty_mean,
                "running_scale_mean": running_scale_mean,
                "raw_sort_ma20_mean": ma20_mean,
                "raw_sort_ma50_mean": ma50_mean,
                "steps_per_action": float(self.steps_per_action),
            }

            info = {
                "interfacial_energy": e.cpu(),
                "motion_penalty": mpen.cpu(),
                "raw_sort_index": raw_sort_idx.cpu(),
                "sort_index": sort_idx.cpu(),
                "delta_sort_index": delta_sort.cpu(),
                "smoothed_delta": self._sort_ema.cpu(),
                "pos_delta": pos_delta.cpu(),
                "pos_delta_rms": self._pos_delta_rms.cpu(),
                "running_scale": running_scale.cpu(),
                "norm_pos_delta": stable_norm_pos_delta.cpu(),
                "sort_term": sort_term.cpu(),
                "bonus_term": bonus_term.cpu(),
                "energy_term": energy_term.cpu(),
                "motion_term": motion_term.cpu(),
                "raw_sort_ma20": self._raw_sort_ma20.cpu(),
                "raw_sort_ma50": self._raw_sort_ma50.cpu(),
                "reward_components": reward_components,
            }

        return self.get_observation(), reward.detach(), info

    # ---------------------------------------------------------
    def current_state(self):
        return self.state.detach().clone()
