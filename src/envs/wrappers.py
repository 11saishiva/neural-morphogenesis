# import torch
# import torch.nn.functional as F
# from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
# from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches


# class SortingEnv:
#     """
#     Cell-sorting environment with 'global' or 'local' observation modes.
#     Local observations: (B, N, C, p, p) where N = H*W.
#     """

#     def __init__(
#         self, H=64, W=64, device='cpu', gamma_motion=0.1,
#         steps_per_action=4, obs_mode='local'
#     ):
#         self.H, self.W = H, W
#         self.device = torch.device(device)
#         self.gamma_motion = gamma_motion
#         self.steps_per_action = steps_per_action
#         self.obs_mode = obs_mode

#         # core simulator
#         self.dca = DCA().to(self.device)
#         self.state = None

#         # -----------------------------
#         # Conservative, robust hyperparams
#         # -----------------------------
#         # Sorting reward ingredients
#         self.sort_weight   = 100.0     # conservative
#         self.sort_bonus    = 0.5       # small immediate bonus
#         self.energy_weight = 1.0
#         self.motion_weight = 0.08      # stronger motion penalty to discourage wiggle
#         self.reward_clip   = 50.0
#         self.term_clip     = 50.0

#         # EMA smoothing (make reactive but not noisy)
#         self.sort_ema_alpha = 0.35
#         self._sort_ema = None
#         self._last_sort_idx = None

#         # RMS normalizer for pos_delta: adapt fairly quickly so norm_pos_delta > 0 appears
#         self.pos_delta_rms_alpha = 0.40
#         self._pos_delta_rms = None
#         self._pos_delta_eps = 1e-6

#         # Moderate amplification (prevents tiny raw_sort exploding)
#         self.SORT_AMPLIFY = 50.0

#         # Diagnostics / moving averages
#         self._raw_sort_ma20 = None
#         self._raw_sort_ma50 = None
#         self._env_step = 0

#     # ---------------------------------------------------------
#     # Helper: morphogen band
#     # ---------------------------------------------------------
#     def _make_morphogen(self, B):
#         x = torch.linspace(0, 1, self.W, device=self.device)
#         x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
#         return x

#     # ---------------------------------------------------------
#     # Reset
#     # ---------------------------------------------------------
#     def reset(self, B=1, pA=0.5):
#         types = torch.rand(B, 2, self.H, self.W, device=self.device)
#         types = F.softmax(types, dim=1)
#         types[:, TYPE_A] = types[:, TYPE_A] * 0.5 + pA
#         types[:, TYPE_B] = types[:, TYPE_B] * 0.5 + (1 - pA)
#         types = F.softmax(types, dim=1)

#         adhesion = torch.rand(B, 1, self.H, self.W, device=self.device) * 0.2 + 0.4
#         morphogen = self._make_morphogen(B)
#         center = torch.ones(B, 1, self.H, self.W, device=self.device)

#         state = torch.cat([types, adhesion, morphogen, center], dim=1)
#         self.state = state.detach().clone()

#         B_actual = state.shape[0]

#         # initialize EMA of delta-sort
#         self._sort_ema = torch.zeros(B_actual, device=self.device)

#         with torch.no_grad():
#             raw = self._sorting_index(self.state).detach()
#             amp = raw * self.SORT_AMPLIFY
#             self._last_sort_idx = amp.clone()

#         # RMS normalizer initialised to 1
#         self._pos_delta_rms = torch.ones(B_actual, device=self.device)

#         # init moving averages
#         self._raw_sort_ma20 = raw.clone()
#         self._raw_sort_ma50 = raw.clone()

#         self._env_step = 0

#         return self.get_observation()

#     # ---------------------------------------------------------
#     def get_observation(self):
#         if self.obs_mode == 'global':
#             return self.state.detach().clone()
#         elif self.obs_mode == 'local':
#             patches, coords = extract_local_patches(
#                 self.state.detach().clone(), patch_size=5
#             )
#             return patches, coords
#         else:
#             raise ValueError("obs_mode must be 'global' or 'local'")

#     # ---------------------------------------------------------
#     # Sorting index: |meanA_left - meanA_right|
#     # ---------------------------------------------------------
#     def _sorting_index(self, state):
#         A = state[:, TYPE_A]  # (B,H,W)
#         mid = self.W // 2
#         left = A[:, :, :mid].mean(dim=[1, 2])
#         right = A[:, :, mid:].mean(dim=[1, 2])
#         return torch.abs(left - right)

#     # ---------------------------------------------------------
#     # Safe scalar extractor
#     # ---------------------------------------------------------
#     @staticmethod
#     def _safe_scalar(x):
#         if isinstance(x, torch.Tensor):
#             return float(x.detach().cpu().mean())
#         return float(x)

#     # ---------------------------------------------------------
#     # Main step
#     # ---------------------------------------------------------
#     def step(self, actions):
#         if self.state is None:
#             raise RuntimeError("Call reset() before step().")

#         B = self.state.shape[0]
#         self._env_step += 1

#         # local action reshape expected: (B, N, A) where N=H*W
#         if self.obs_mode == 'local':
#             N = self.H * self.W
#             if actions.dim() != 3 or actions.shape[1] != N:
#                 raise ValueError(f"Expected (B, N, A). Got {actions.shape}")
#             actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)

#         with torch.no_grad():
#             # apply DCA for several micro-steps
#             s = self.state
#             for _ in range(self.steps_per_action):
#                 s = self.dca(s, actions, steps=1)
#             self.state = s.detach().clone()

#             # energy and motion
#             e = interfacial_energy(self.state).detach()           # (B,)
#             mpen = motion_penalty(actions.detach()).detach()      # (B,)

#             # sorting index
#             raw_sort_idx = self._sorting_index(self.state).detach()  # (B,)
#             sort_idx = raw_sort_idx * self.SORT_AMPLIFY               # (B,)

#             # delta sort
#             if self._last_sort_idx is None:
#                 delta_sort = torch.zeros_like(sort_idx)
#             else:
#                 delta_sort = sort_idx - self._last_sort_idx
#             self._last_sort_idx = sort_idx.detach().clone()

#             # EMA of delta-sort (reactive)
#             if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
#                 self._sort_ema = torch.zeros_like(sort_idx)
#             α = float(self.sort_ema_alpha)
#             self._sort_ema = (1 - α) * self._sort_ema.to(sort_idx.device) + α * delta_sort
#             pos_delta = torch.relu(self._sort_ema)

#             # RMS normalization of positive delta
#             if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
#                 self._pos_delta_rms = torch.ones_like(pos_delta)
#             β = float(self.pos_delta_rms_alpha)
#             sq = pos_delta ** 2
#             self._pos_delta_rms = (1 - β) * self._pos_delta_rms.to(pos_delta.device) + β * sq
#             running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

#             denom = running_scale + self._pos_delta_eps
#             norm_pos_delta = pos_delta / denom
#             norm_pos_delta = torch.nan_to_num(norm_pos_delta, nan=0.0, posinf=0.0, neginf=0.0)

#             # moving averages for diagnostics
#             if (self._raw_sort_ma20 is None) or (self._raw_sort_ma20.shape[0] != raw_sort_idx.shape[0]):
#                 self._raw_sort_ma20 = raw_sort_idx.clone()
#                 self._raw_sort_ma50 = raw_sort_idx.clone()
#             else:
#                 alpha20 = 2.0 / (20.0 + 1.0)
#                 alpha50 = 2.0 / (50.0 + 1.0)
#                 self._raw_sort_ma20 = (1.0 - alpha20) * self._raw_sort_ma20 + alpha20 * raw_sort_idx
#                 self._raw_sort_ma50 = (1.0 - alpha50) * self._raw_sort_ma50 + alpha50 * raw_sort_idx

#             # -------------------------
#             # Reward terms (bounded)
#             # -------------------------
#             # Bound normalized delta with tanh so small noise can't explode under amplification.
#             bounded_norm = torch.tanh(norm_pos_delta * 8.0)  # maps small normalized values into smooth region
#             sort_term = self.sort_weight * bounded_norm
#             bonus_term = self.sort_bonus * torch.tanh(sort_idx)    # bounded immediate bonus
#             energy_term = -self.energy_weight * e
#             motion_term = -self.motion_weight * mpen

#             # Per-term clipping for stability
#             sort_term = torch.clamp(sort_term, -self.term_clip, self.term_clip)
#             bonus_term = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
#             energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
#             motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

#             reward = sort_term + bonus_term + energy_term + motion_term
#             reward = reward / float(self.steps_per_action)
#             reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

#             # -------------------------
#             # Diagnostics and logging
#             # -------------------------
#             raw_mean = self._safe_scalar(raw_sort_idx)
#             ma20_mean = self._safe_scalar(self._raw_sort_ma20)
#             ma50_mean = self._safe_scalar(self._raw_sort_ma50)
#             reward_mean = self._safe_scalar(reward)
#             norm_pos_delta_mean = self._safe_scalar(norm_pos_delta)

#             if (self._env_step % 10) == 0:
#                 print(
#                     f"[SORT-MA] step={self._env_step} raw={raw_mean:.6e} ma20={ma20_mean:.6e} "
#                     f"ma50={ma50_mean:.6e} reward_mean={reward_mean:.6f} norm_pos_delta={norm_pos_delta_mean:.6e}",
#                     flush=True,
#                 )

#             # pack scalar components
#             reward_components = {
#                 "reward_mean": reward_mean,
#                 "sort_term_mean": self._safe_scalar(sort_term),
#                 "bonus_term_mean": self._safe_scalar(bonus_term),
#                 "energy_term_mean": self._safe_scalar(energy_term),
#                 "motion_term_mean": self._safe_scalar(motion_term),
#                 "raw_sort_idx_mean": raw_mean,
#                 "sort_idx_mean": self._safe_scalar(sort_idx),
#                 "pos_delta_mean": self._safe_scalar(pos_delta),
#                 "norm_pos_delta_mean": norm_pos_delta_mean,
#                 "interfacial_energy_mean": self._safe_scalar(e),
#                 "motion_penalty_mean": self._safe_scalar(mpen),
#                 "running_scale_mean": self._safe_scalar(running_scale),
#                 "raw_sort_ma20_mean": ma20_mean,
#                 "raw_sort_ma50_mean": ma50_mean,
#                 "steps_per_action": float(self.steps_per_action),
#             }

#             # info: prefer tensors (for debugging) but safe conversion wrapper
#             def _maybe_cpu(x):
#                 return x.detach().cpu() if isinstance(x, torch.Tensor) else x

#             info = {
#                 "interfacial_energy": _maybe_cpu(e),
#                 "motion_penalty": _maybe_cpu(mpen),
#                 "raw_sort_index": _maybe_cpu(raw_sort_idx),
#                 "sort_index": _maybe_cpu(sort_idx),
#                 "delta_sort_index": _maybe_cpu(delta_sort),
#                 "smoothed_delta": _maybe_cpu(self._sort_ema),
#                 "pos_delta": _maybe_cpu(pos_delta),
#                 "norm_pos_delta": _maybe_cpu(norm_pos_delta),
#                 "pos_delta_rms": _maybe_cpu(self._pos_delta_rms),
#                 "running_scale": _maybe_cpu(running_scale),
#                 "sort_term": _maybe_cpu(sort_term),
#                 "bonus_term": _maybe_cpu(bonus_term),
#                 "energy_term": _maybe_cpu(energy_term),
#                 "motion_term": _maybe_cpu(motion_term),
#                 "raw_sort_ma20": _maybe_cpu(self._raw_sort_ma20),
#                 "raw_sort_ma50": _maybe_cpu(self._raw_sort_ma50),
#                 "reward_components": reward_components,
#             }

#         return self.get_observation(), reward.detach(), info

#     # ---------------------------------------------------------
#     def current_state(self):
#         return self.state.detach().clone()

import torch
import torch.nn.functional as F
from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches


class SortingEnv:
    """
    Cell-sorting environment with 'global' or 'local' observation modes.
    Local observations: (B, N, C, p, p) where N = H*W.

    Notes:
    - Actions expected:
      * obs_mode == 'local'  -> actions shape (B, N, A) where A == 3 (adh, vx, vy)
      * obs_mode == 'global' -> actions shape (B, 3, H, W)
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

        # --- Reward shaping coefficients (final tuned defaults) ---
        self.sort_weight   = 200.0    # main multiplier for sort-related reward (kept moderate)
        self.sort_bonus    = 2.0
        self.energy_weight = 1.0
        self.motion_weight = 0.03
        self.reward_clip   = 50.0
        self.term_clip     = 50.0

        # EMA smoothing of delta_sort
        self.sort_ema_alpha = 0.4
        self._sort_ema = None
        self._last_sort_idx = None

        # RMS normalizer for pos_delta
        self.pos_delta_rms_alpha = 0.20
        self._pos_delta_rms = None
        self._pos_delta_eps = 1e-6

        # Scaling factor for sorting index (kept moderate)
        self.SORT_AMPLIFY = 200.0

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
        # initialize types with soft probabilities biased by pA
        types = torch.rand(B, 2, self.H, self.W, device=self.device)
        types = F.softmax(types, dim=1)
        types[:, TYPE_A] = types[:, TYPE_A] * 0.5 + pA
        types[:, TYPE_B] = types[:, TYPE_B] * 0.5 + (1 - pA)
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

        # initial sorting index baseline
        with torch.no_grad():
            raw = self._sorting_index(self.state).detach()
            self._last_sort_idx = (raw * self.SORT_AMPLIFY).clone()
            # initialize moving averages
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
        A = state[:, TYPE_A]  # (B,H,W)
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
            # actions must be (B, N, A) where N = H*W
            if not (actions.dim() == 3 and actions.shape[0] == B):
                raise ValueError(f"For local mode expected actions (B, N, A); got {actions.shape}")
            N = self.H * self.W
            if actions.shape[1] != N:
                raise ValueError(f"Local actions second dim must be H*W={N}; got {actions.shape[1]}")
            # transpose to (B, A, H, W) for DCA
            actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)
        else:
            # global mode: expect (B, 3, H, W)
            if not (actions.dim() == 4 and actions.shape == (B, 3, self.H, self.W)):
                raise ValueError(f"For global mode expected (B,3,H,W). Got {actions.shape}")

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
            raw_sort_idx = self._sorting_index(self.state).detach()  # tensor (B,)
            sort_idx = raw_sort_idx * self.SORT_AMPLIFY

            if self._last_sort_idx is None:
                delta_sort = torch.zeros_like(sort_idx)
            else:
                delta_sort = sort_idx - self._last_sort_idx

            self._last_sort_idx = sort_idx.detach().clone()

            # EMA of delta-sort
            if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
                self._sort_ema = torch.zeros_like(sort_idx)

            alpha = float(self.sort_ema_alpha)
            self._sort_ema = (1 - alpha) * self._sort_ema.to(sort_idx.device) + alpha * delta_sort

            pos_delta = torch.relu(self._sort_ema)  # tensor (B,)

            # RMS normalization of pos_delta (running variance-like)
            if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
                self._pos_delta_rms = torch.ones_like(pos_delta)

            beta = float(self.pos_delta_rms_alpha)
            sq = pos_delta ** 2
            self._pos_delta_rms = (1 - beta) * self._pos_delta_rms.to(pos_delta.device) + beta * sq
            running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

            norm_pos_delta = pos_delta / (running_scale + self._pos_delta_eps)  # tensor (B,)

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
            sort_term   = self.sort_weight * norm_pos_delta
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
                # Accepts: tensor, numpy scalar, python float
                if isinstance(x, torch.Tensor):
                    return float(x.detach().cpu().mean())
                try:
                    return float(x)
                except Exception:
                    # fallback safe conversion
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
            norm_pos_delta_mean = _scalar(norm_pos_delta)
            running_scale_mean = _scalar(running_scale)
            interfacial_energy_mean = _scalar(e)
            motion_penalty_mean = _scalar(mpen)

            # periodic minimal logging
            if (self._env_step % 10) == 0:
                print(
                    f"[SORT-MA] step={self._env_step} "
                    f"raw={raw_mean:.6e} ma20={ma20_mean:.6e} ma50={ma50_mean:.6e} "
                    f"reward_mean={reward_mean:.6f} norm_pos_delta={norm_pos_delta_mean:.6e}",
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

            # Full tensor info for debugging (returned as CPU tensors)
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
                "norm_pos_delta": norm_pos_delta.cpu(),
                "sort_term": sort_term.cpu(),
                "bonus_term": bonus_term.cpu(),
                "energy_term": energy_term.cpu(),
                "motion_term": motion_term.cpu(),
                "raw_sort_ma20": self._raw_sort_ma20.cpu(),
                "raw_sort_ma50": self._raw_sort_ma50.cpu(),
                "reward_components": reward_components,
            }

        # Return (obs, reward tensor, info dict)
        return self.get_observation(), reward.detach(), info

    # ---------------------------------------------------------
    def current_state(self):
        return self.state.detach().clone()
