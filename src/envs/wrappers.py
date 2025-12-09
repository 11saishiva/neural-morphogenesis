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
#         steps_per_action=6, obs_mode='local'
#     ):
#         self.H, self.W = H, W
#         self.device = torch.device(device)
#         self.gamma_motion = gamma_motion
#         self.steps_per_action = steps_per_action
#         self.obs_mode = obs_mode

#         self.dca = DCA().to(self.device)
#         self.state = None

#         # --- Reward shaping coefficients: “research-clean” defaults ---
#         self.sort_weight   = 1600.0
#         self.sort_bonus    = 40.0
#         self.energy_weight = 1.0
#         self.motion_weight = 0.03
#         self.reward_clip   = 50.0
#         self.term_clip     = 50.0

#         # EMA smoothing of delta_sort
#         self.sort_ema_alpha = 0.2
#         self._sort_ema = None
#         self._last_sort_idx = None

#         # RMS normalizer for pos_delta
#         self.pos_delta_rms_alpha = 0.05
#         self._pos_delta_rms = None
#         self._pos_delta_eps = 1e-6

#         # Critical: scaling factor for sorting index
#         self.SORT_AMPLIFY = 5000.0   # your current research-default


#     # ---------------------------------------------------------
#     # Helper: create a morphogen band
#     # ---------------------------------------------------------
#     def _make_morphogen(self, B):
#         x = torch.linspace(0, 1, self.W, device=self.device)
#         x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
#         return x


#     # ---------------------------------------------------------
#     # Reset environment
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
#         self._pos_delta_rms = torch.ones(B_actual, device=self.device) * 1.0

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
#     # Sorting index = difference of mean A on left vs right half
#     # ---------------------------------------------------------
#     def _sorting_index(self, state):
#         A = state[:, TYPE_A]  # (B,H,W)
#         mid = self.W // 2
#         left = A[:, :, :mid].mean(dim=[1, 2])
#         right = A[:, :, mid:].mean(dim=[1, 2])
#         return torch.abs(left - right)


#     # ---------------------------------------------------------
#     # MAIN STEP FUNCTION
#     # ---------------------------------------------------------
#     def step(self, actions):
#         if self.state is None:
#             raise RuntimeError("Call reset() before step().")

#         B = self.state.shape[0]

#         # reshape local actions
#         if self.obs_mode == 'local':
#             N = self.H * self.W
#             if actions.dim() != 3 or actions.shape[1] != N:
#                 raise ValueError(f"Expected (B, N, A). Got {actions.shape}")
#             actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)

#         with torch.no_grad():
#             # -------------------
#             # Apply DCA for k steps
#             # -------------------
#             s = self.state
#             for _ in range(self.steps_per_action):
#                 s = self.dca(s, actions, steps=1)

#             self.state = s.detach().clone()

#             # Compute energy + motion
#             e = interfacial_energy(self.state).detach()           # (B,)
#             mpen = motion_penalty(actions.detach()).detach()      # (B,)

#             # Sorting index
#             raw_sort_idx = self._sorting_index(self.state).detach()
#             sort_idx = raw_sort_idx * self.SORT_AMPLIFY

#             if self._last_sort_idx is None:
#                 delta_sort = torch.zeros_like(sort_idx)
#             else:
#                 delta_sort = sort_idx - self._last_sort_idx

#             self._last_sort_idx = sort_idx.detach().clone()

#             # EMA of delta-sort
#             if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
#                 self._sort_ema = torch.zeros_like(sort_idx)

#             α = float(self.sort_ema_alpha)
#             self._sort_ema = (1 - α) * self._sort_ema.to(sort_idx.device) + α * delta_sort

#             pos_delta = torch.relu(self._sort_ema)

#             # RMS norm
#             if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
#                 self._pos_delta_rms = torch.ones_like(pos_delta)

#             β = float(self.pos_delta_rms_alpha)
#             sq = pos_delta ** 2
#             self._pos_delta_rms = (1 - β) * self._pos_delta_rms.to(pos_delta.device) + β * sq
#             running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

#             norm_pos_delta = pos_delta / (running_scale + self._pos_delta_eps)

#             # -----------------------------------------------------
#             # REWARD TERMS (unchanged, but now we will add diagnostics)
#             # -----------------------------------------------------
#             sort_term  = self.sort_weight * norm_pos_delta
#             bonus_term = self.sort_bonus * raw_sort_idx
#             energy_term = -self.energy_weight * e
#             motion_term = -self.motion_weight * mpen

#             # Clip per-term
#             sort_term   = torch.clamp(sort_term, -self.term_clip, self.term_clip)
#             bonus_term  = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
#             energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
#             motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

#             reward = sort_term + bonus_term + energy_term + motion_term
#             reward = reward / float(self.steps_per_action)
#             reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

#             # -----------------------------------------------------
#             # NEW DIAGNOSTICS (Action A)
#             # -----------------------------------------------------
#             def _scalar(x):
#                 return float(x.detach().cpu().mean())

#             reward_components = {
#                 "reward_mean": _scalar(reward),
#                 "sort_term_mean": _scalar(sort_term),
#                 "bonus_term_mean": _scalar(bonus_term),
#                 "energy_term_mean": _scalar(energy_term),
#                 "motion_term_mean": _scalar(motion_term),
#                 "raw_sort_idx_mean": _scalar(raw_sort_idx),
#                 "sort_idx_mean": _scalar(sort_idx),
#                 "pos_delta_mean": _scalar(pos_delta),
#                 "interfacial_energy_mean": _scalar(e),
#                 "motion_penalty_mean": _scalar(mpen),
#                 "running_scale_mean": _scalar(running_scale),
#                 "steps_per_action": float(self.steps_per_action),
#             }

#             # Full tensor info for debugging
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
#                 "norm_pos_delta": norm_pos_delta.cpu(),
#                 "sort_term": sort_term.cpu(),
#                 "bonus_term": bonus_term.cpu(),
#                 "energy_term": energy_term.cpu(),
#                 "motion_term": motion_term.cpu(),
#                 "reward_components": reward_components,   # NEW
#             }

#         return self.get_observation(), reward.detach(), info


#     # ---------------------------------------------------------
#     def current_state(self):
#         return self.state.detach().clone()

# wrappers.py
import torch
import torch.nn.functional as F
from .dca import DCA, TYPE_A, TYPE_B, ADH, MORPH, CENTER
from ..utils.metrics import interfacial_energy, motion_penalty, extract_local_patches

"""
wrappers.py - single-file environment with integrated, safe applied-sort logic.

This file:
 - Keeps your original reward terms (sort_term, bonus_term, energy_term, motion_term)
 - Adds an optional 'applied_sort' contribution computed from raw_sort_index using:
     amplification, running_scale, gating, soft/hard clipping, and a visible-test override.
 - No external files created. Configure the new behavior via attributes on SortingEnv.

How to toggle:
 - self.USE_APPLIED_SORT = True/False
 - self.APPLIED_SORT_CFG contains parameters (SORT_AMPLIFY, term_clip, zero_gate_threshold, ...)
"""

class SortingEnv:
    """
    Cell-sorting environment with 'global' or 'local' observation modes.
    Local observations: (B, N, C, p, p) where N = H*W.
    """

    def __init__(
        self, H=64, W=64, device='cpu', gamma_motion=0.1,
        steps_per_action=12, obs_mode='local'
    ):
        self.H, self.W = H, W
        self.device = torch.device(device)
        self.gamma_motion = gamma_motion
        self.steps_per_action = steps_per_action
        self.obs_mode = obs_mode

        self.dca = DCA().to(self.device)
        self.state = None

        # --- Reward shaping coefficients: “research-clean” defaults ---
        self.sort_weight   = 1600.0
        self.sort_bonus    = 40.0
        self.energy_weight = 1.0
        self.motion_weight = 0.03
        self.reward_clip   = 50.0
        self.term_clip     = 50.0

        # EMA smoothing of delta_sort
        self.sort_ema_alpha = 0.5
        self._sort_ema = None
        self._last_sort_idx = None

        # RMS normalizer for pos_delta
        self.pos_delta_rms_alpha = 0.01
        self._pos_delta_rms = None
        self._pos_delta_eps = 1e-6

        # Critical: scaling factor for sorting index
        self.SORT_AMPLIFY = 15000.0   # your current research-default

        # --- New integrated applied-sort settings ---
        # Toggle to enable/disable this additional applied_sort term.
        self.USE_APPLIED_SORT = True

        # Per-env config used by compute_applied_sort_tensor
        self.APPLIED_SORT_CFG = {
            # multiplier applied to raw_sort_index first (matches research default)
            "SORT_AMPLIFY": float(self.SORT_AMPLIFY),
            # clipping magnitude applied after running-scale multiplication (None disables)
            "term_clip": float(0.05),    # default small soft clip; adjust as needed
            # threshold on abs(pre_scaled) below which the term is zeroed (None disables)
            "zero_gate_threshold": 1e-4,
            # soft clip (True) or hard clip (False)
            "use_soft_clip": True,
            # visible test override: set float value to force applied_sort to that value (None to disable)
            "visible_test_value": None,
            # bypass gating entirely (useful for quick tests)
            "bypass_gating": False,
            # protect against running_scale == 0
            "min_running_scale": 1e-8,
            # If you want to weight the applied_sort term separately from existing sort_weight/bonus:
            "applied_sort_weight": 1.0,
        }

    # ---------------------------------------------------------
    # Helper: create a morphogen band
    # ---------------------------------------------------------
    def _make_morphogen(self, B):
        x = torch.linspace(0, 1, self.W, device=self.device)
        x = x.view(1, 1, 1, self.W).repeat(B, 1, self.H, 1)
        return x

    # ---------------------------------------------------------
    # Reset environment
    # ---------------------------------------------------------
    def reset(self, B=1, pA=0.5):
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

        # initialize EMA of delta-sort
        self._sort_ema = torch.zeros(B_actual, device=self.device)

        with torch.no_grad():
            raw = self._sorting_index(self.state).detach()
            amp = raw * self.SORT_AMPLIFY
            self._last_sort_idx = amp.clone()

        # RMS normalizer initialised to 1
        self._pos_delta_rms = torch.ones(B_actual, device=self.device) * 1.0

        return self.get_observation()

    # ---------------------------------------------------------
    def get_observation(self):
        if self.obs_mode == 'global':
            return self.state.detach().clone()
        elif self.obs_mode == 'local':
            patches, coords = extract_local_patches(
                self.state.detach().clone(), patch_size=5
            )
            return patches, coords
        else:
            raise ValueError("obs_mode must be 'global' or 'local'")

    # ---------------------------------------------------------
    # Sorting index = difference of mean A on left vs right half
    # ---------------------------------------------------------
    def _sorting_index(self, state):
        A = state[:, TYPE_A]  # (B,H,W)
        mid = self.W // 2
        left = A[:, :, :mid].mean(dim=[1, 2])
        right = A[:, :, mid:].mean(dim=[1, 2])
        return torch.abs(left - right)

    # ---------------------------------------------------------
    # Internal helper: compute applied sort term (torch-friendly)
    # ---------------------------------------------------------
    def compute_applied_sort_tensor(self, raw_sort_idx, running_scale, cfg=None, debug=False):
        """
        raw_sort_idx: (B,) tensor (small values like 1e-5)
        running_scale: (B,) tensor (positive)
        cfg: dict overriding APPLIED_SORT_CFG keys
        Returns:
            applied_sort: (B,) tensor
            diagnostics: dict with small cpu tensors/scalars useful for info logging
        """

        # Merge cfg
        cfg_use = dict(self.APPLIED_SORT_CFG)
        if cfg:
            cfg_use.update(cfg)

        # Safe casts
        ampl = float(cfg_use.get("SORT_AMPLIFY", float(self.SORT_AMPLIFY)))
        term_clip = cfg_use.get("term_clip", None)
        zero_gate_threshold = cfg_use.get("zero_gate_threshold", None)
        use_soft_clip = bool(cfg_use.get("use_soft_clip", True))
        visible_test_value = cfg_use.get("visible_test_value", None)
        bypass_gating = bool(cfg_use.get("bypass_gating", False))
        min_running_scale = float(cfg_use.get("min_running_scale", 1e-8))
        applied_sort_weight = float(cfg_use.get("applied_sort_weight", 1.0))

        # ensure running_scale is safe
        running_scale_safe = torch.clamp(running_scale, min=min_running_scale)

        # pre-scale
        pre_scaled = raw_sort_idx * ampl              # (B,)

        # scaled by running scale (elementwise)
        scaled_by_running = pre_scaled * running_scale_safe  # (B,)

        applied = scaled_by_running.clone()

        gating_reason = None
        # apply gating if configured (compare abs(pre_scaled) to threshold)
        if (not bypass_gating) and (zero_gate_threshold is not None):
            try:
                zt = float(zero_gate_threshold)
                # zero where abs(pre_scaled) < zt
                mask_zero = (pre_scaled.abs() < zt)
                if mask_zero.any():
                    applied = applied.masked_fill(mask_zero, 0.0)
                    # store simple gating reason summary
                    gating_reason = f"zeroed_when_abs(pre_scaled) < {zt}"
            except Exception:
                # treat as misconfigured -> do nothing, record reason
                gating_reason = "zero_gate_threshold_misconfigured"

        clip_mode = None
        clip_denom = None
        # apply clip (soft or hard) if configured and applied != 0
        if (term_clip is not None):
            try:
                clip_val = float(term_clip)
                if clip_val > 0.0:
                    if use_soft_clip:
                        # soft clip: x -> x / (1 + |x|/clip_val)
                        # compute denom per-element
                        denom = 1.0 + (applied.abs() / clip_val)
                        # avoid divide by zero problems
                        applied = applied / denom
                        clip_mode = "soft"
                        # store a cpu tensor of denom for diagnostics if desired
                        clip_denom = denom.detach().cpu()
                    else:
                        # hard clip
                        applied = torch.clamp(applied, -clip_val, clip_val)
                        clip_mode = "hard"
            except Exception:
                clip_mode = "clip_misconfigured"

        # visible test override (force applied to a fixed scalar)
        if visible_test_value is not None:
            val = float(visible_test_value)
            applied = torch.ones_like(applied) * val

        # final weight scaling (if you want applied sort to be weighted separately)
        applied = applied * applied_sort_weight

        # diagnostics (move small tensors to cpu)
        diagnostics = {
            "raw_sort_idx_cpu": raw_sort_idx.detach().cpu(),
            "pre_scaled_cpu": pre_scaled.detach().cpu(),
            "scaled_by_running_cpu": scaled_by_running.detach().cpu(),
            "applied_sort_cpu": applied.detach().cpu(),
            "gating_reason": gating_reason,
            "clip_mode": clip_mode,
        }

        if debug:
            print("compute_applied_sort_tensor diag:")
            print(" pre_scaled (cpu):", diagnostics["pre_scaled_cpu"].mean().item())
            print(" scaled_by_running (cpu):", diagnostics["scaled_by_running_cpu"].mean().item())
            print(" applied_sort (cpu):", diagnostics["applied_sort_cpu"].mean().item())
            if gating_reason:
                print(" gating:", gating_reason)
            if clip_mode:
                print(" clip_mode:", clip_mode)

        return applied, diagnostics

    # ---------------------------------------------------------
    # MAIN STEP FUNCTION
    # ---------------------------------------------------------
    def step(self, actions):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        B = self.state.shape[0]

        # reshape local actions
        if self.obs_mode == 'local':
            N = self.H * self.W
            if actions.dim() != 3 or actions.shape[1] != N:
                raise ValueError(f"Expected (B, N, A). Got {actions.shape}")
            actions = actions.transpose(1, 2).reshape(B, -1, self.H, self.W)

        with torch.no_grad():
            # -------------------
            # Apply DCA for k steps
            # -------------------
            s = self.state
            for _ in range(self.steps_per_action):
                s = self.dca(s, actions, steps=1)

            self.state = s.detach().clone()

            # Compute energy + motion
            e = interfacial_energy(self.state).detach()           # (B,)
            mpen = motion_penalty(actions.detach()).detach()      # (B,)

            # Sorting index
            raw_sort_idx = self._sorting_index(self.state).detach()  # (B,)
            sort_idx = raw_sort_idx * self.SORT_AMPLIFY            # (B,)

            if self._last_sort_idx is None:
                delta_sort = torch.zeros_like(sort_idx)
            else:
                delta_sort = sort_idx - self._last_sort_idx

            self._last_sort_idx = sort_idx.detach().clone()

            # EMA of delta-sort
            if self._sort_ema is None or self._sort_ema.shape[0] != sort_idx.shape[0]:
                self._sort_ema = torch.zeros_like(sort_idx)

            α = float(self.sort_ema_alpha)
            self._sort_ema = (1 - α) * self._sort_ema.to(sort_idx.device) + α * delta_sort

            pos_delta = torch.relu(self._sort_ema)

            # RMS norm
            if self._pos_delta_rms is None or self._pos_delta_rms.shape[0] != pos_delta.shape[0]:
                self._pos_delta_rms = torch.ones_like(pos_delta)

            β = float(self.pos_delta_rms_alpha)
            sq = pos_delta ** 2
            self._pos_delta_rms = (1 - β) * self._pos_delta_rms.to(pos_delta.device) + β * sq
            running_scale = torch.sqrt(self._pos_delta_rms + self._pos_delta_eps)

            norm_pos_delta = pos_delta / (running_scale + self._pos_delta_eps)

            # -----------------------------------------------------
            # REWARD TERMS (unchanged)
            # -----------------------------------------------------
            sort_term  = self.sort_weight * norm_pos_delta
            bonus_term = self.sort_bonus * raw_sort_idx
            energy_term = -self.energy_weight * e
            motion_term = -self.motion_weight * mpen

            # Clip per-term
            sort_term   = torch.clamp(sort_term, -self.term_clip, self.term_clip)
            bonus_term  = torch.clamp(bonus_term, -self.term_clip, self.term_clip)
            energy_term = torch.clamp(energy_term, -self.term_clip, self.term_clip)
            motion_term = torch.clamp(motion_term, -self.term_clip, self.term_clip)

            reward = sort_term + bonus_term + energy_term + motion_term

            # ---------------------------
            # NEW: compute applied_sort and add it (optional)
            # ---------------------------
            applied_sort_tensor = torch.zeros_like(reward)
            applied_sort_diag = None
            if self.USE_APPLIED_SORT:
                # compute per-batch applied_sort using raw_sort_idx and running_scale
                # pass a cfg copy so call-sites can modify without altering env default
                cfg_local = dict(self.APPLIED_SORT_CFG)
                # ensure the amplitude tracks the env SORT_AMPLIFY
                cfg_local["SORT_AMPLIFY"] = float(self.SORT_AMPLIFY)
                # compute applied sort (B,) tensor
                applied_sort_tensor, applied_sort_diag = self.compute_applied_sort_tensor(
                    raw_sort_idx, running_scale, cfg=cfg_local, debug=False
                )
                # applied_sort_tensor is (B,) — expand to match reward shape (B,)
                # (reward shape is (B,), so direct addition is fine)
                # add the term (it was scaled inside compute by applied_sort_weight already)
                reward = reward + applied_sort_tensor

            # finish reward normalization across steps and clip
            reward = reward / float(self.steps_per_action)
            reward = torch.clamp(reward, -self.reward_clip, self.reward_clip)

            # -----------------------------------------------------
            # NEW DIAGNOSTICS (Action A)
            # -----------------------------------------------------
            def _scalar(x):
                # accepts tensor or float
                if torch.is_tensor(x):
                    return float(x.detach().cpu().mean())
                return float(x)

            reward_components = {
                "reward_mean": _scalar(reward),
                "sort_term_mean": _scalar(sort_term),
                "bonus_term_mean": _scalar(bonus_term),
                "energy_term_mean": _scalar(energy_term),
                "motion_term_mean": _scalar(motion_term),
                "raw_sort_idx_mean": _scalar(raw_sort_idx),
                "sort_idx_mean": _scalar(sort_idx),
                "pos_delta_mean": _scalar(pos_delta),
                "interfacial_energy_mean": _scalar(e),
                "motion_penalty_mean": _scalar(mpen),
                "running_scale_mean": _scalar(running_scale),
                "steps_per_action": float(self.steps_per_action),
                "use_applied_sort": bool(self.USE_APPLIED_SORT),
            }

            if self.USE_APPLIED_SORT:
                reward_components["applied_sort_mean"] = _scalar(applied_sort_tensor)
                # expose a simple scalar of the amplifier & gate settings used
                reward_components["applied_sort_cfg"] = dict(self.APPLIED_SORT_CFG)

            # Full tensor info for debugging
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
                "reward_components": reward_components,   # NEW
            }

            if self.USE_APPLIED_SORT:
                info["applied_sort"] = applied_sort_tensor.cpu()
                if applied_sort_diag is not None:
                    # include pre_scaled and other small diag arrays from compute_applied_sort_tensor
                    info["applied_sort_diag"] = applied_sort_diag

        return self.get_observation(), reward.detach(), info

    # ---------------------------------------------------------
    def current_state(self):
        return self.state.detach().clone()
