# # src/utils/metrics.py
# import torch
# import torch.nn.functional as F

# def extract_local_patches(state: torch.Tensor, patch_size: int = 5):
#     """
#     Safe patch extractor (no as_strided).
#     Input:
#         state: (B, C, H, W)
#     Output:
#         patches: (B, H*W, C, patch_size, patch_size)  -- contiguous
#         coords:  (B, H*W, 2)                           -- int coords (y,x)
#     """
#     if state.dim() != 4:
#         raise ValueError("state must be (B,C,H,W)")

#     B, C, H, W = state.shape
#     p = patch_size
#     pad = p // 2

#     # pad to allow extracting patches at borders
#     padded = F.pad(state, (pad, pad, pad, pad), mode="reflect")  # (B, C, H+2pad, W+2pad)

#     patches_list = []
#     coords_list = []

#     # iterate spatially (H small enough typically)
#     for y in range(H):
#         for x in range(W):
#             # slice returns contiguous only if we clone; clone -> contiguous copy
#             patch = padded[:, :, y:(y+p), x:(x+p)].clone().contiguous()  # (B, C, p, p)
#             patches_list.append(patch)
#             coords_list.append(torch.tensor([y, x], device=state.device, dtype=torch.long))

#     # stack patches: list length = N (=H*W), each element (B,C,p,p)
#     # after stacking: (N, B, C, p, p) -> permute to (B, N, C, p, p)
#     patches_stack = torch.stack(patches_list, dim=0)        # (N, B, C, p, p)
#     patches_stack = patches_stack.permute(1, 0, 2, 3, 4)   # (B, N, C, p, p)
#     patches_stack = patches_stack.contiguous()

#     coords = torch.stack(coords_list, dim=0)               # (N, 2)
#     coords = coords.unsqueeze(0).repeat(B, 1, 1)           # (B, N, 2)

#     return patches_stack, coords


# def interfacial_energy(state: torch.Tensor, axis_pairs=None):
#     """
#     Simple interfacial energy estimator.
#     Input:
#         state: (B, C, H, W) where channels include TYPE_A and TYPE_B probabilities
#     Output:
#         energy: (B,) per-batch scalar energy (mean magnitude of gradients between types)
#     Notes:
#         - This is a generic, stable proxy for interfacial energy: spatial gradient magnitude
#           of the difference in type probability fields.
#     """
#     if state.dim() != 4:
#         raise ValueError("state must be (B,C,H,W)")

#     # If channels >=2, assume channel 0 corresponds to TYPE_A (compatible with your wrappers).
#     # Use difference between channel 0 and 1 if present, else use channel 0 alone.
#     B, C, H, W = state.shape
#     if C >= 2:
#         field = state[:, 0] - state[:, 1]  # (B, H, W)
#     else:
#         field = state[:, 0]

#     # compute gradients (finite differences)
#     # pad=False: compute internal diffs, then compute absolute gradients
#     dx = torch.abs(field[:, :, 1:] - field[:, :, :-1])   # (B, H, W-1)
#     dy = torch.abs(field[:, 1:, :] - field[:, :-1, :])   # (B, H-1, W)

#     # average gradient magnitude across space -> energy per batch
#     # normalize by number of comparisons to keep scale stable across sizes
#     gx = dx.mean(dim=[1,2]) if dx.numel() > 0 else torch.zeros(B, device=state.device)
#     gy = dy.mean(dim=[1,2]) if dy.numel() > 0 else torch.zeros(B, device=state.device)

#     energy = (gx + gy) * 0.5
#     return energy


# def motion_penalty(actions: torch.Tensor, kind: str = "l2"):
#     """
#     Motion penalty for actions.
#     Input:
#         actions: either
#             - (B, A, H, W)  (global)
#             - (B, N, A)     (flattened local per-cell actions)
#             - (B, A, N)     (some variants)
#     Output:
#         penalty: (B,) per-batch scalar (mean squared magnitude)
#     """
#     if not isinstance(actions, torch.Tensor):
#         actions = torch.tensor(actions)

#     # normalize shape into (B, A, H, W) or (B, N, A)
#     if actions.dim() == 4:
#         # (B, A, H, W)
#         mag = actions.pow(2).mean(dim=[1,2,3])   # per-batch mean squared action
#     elif actions.dim() == 3:
#         # ambiguous: (B, N, A) or (B, A, N). Detect by size of last dim
#         B, x, z = actions.shape
#         # if last dim small (<=8) assume (B, N, A); else try to be robust
#         if z <= 8:
#             mag = actions.pow(2).mean(dim=[1,2])  # (B,)
#         else:
#             # treat as (B, A, N)
#             mag = actions.pow(2).mean(dim=[1,2])
#     elif actions.dim() == 2:
#         # (B, A) or (B, N)
#         mag = actions.pow(2).mean(dim=1)
#     else:
#         # fallback: compute global mean and broadcast
#         mag = actions.pow(2).mean().view(1).repeat(actions.shape[0] if actions.dim() > 0 else 1)

#     return mag

import torch
import torch.nn.functional as F

def extract_local_patches(state: torch.Tensor, patch_size: int = 5):
    """
    Safe patch extractor (no as_strided).
    Input:
        state: (B, C, H, W)
    Output:
        patches: (B, H*W, C, patch_size, patch_size)  -- contiguous
        coords:  (B, H*W, 2)                           -- int coords (y,x)
    """
    if state.dim() != 4:
        raise ValueError("state must be (B,C,H,W)")

    B, C, H, W = state.shape
    p = patch_size
    pad = p // 2

    # pad to allow extracting patches at borders
    padded = F.pad(state, (pad, pad, pad, pad), mode="reflect")  # (B, C, H+2pad, W+2pad)

    patches_list = []
    coords_list = []

    # iterate spatially (H small enough typically)
    for y in range(H):
        for x in range(W):
            # slice returns contiguous only if we clone; clone -> contiguous copy
            patch = padded[:, :, y:(y+p), x:(x+p)].clone().contiguous()  # (B, C, p, p)
            patches_list.append(patch)
            coords_list.append(torch.tensor([y, x], device=state.device, dtype=torch.long))

    # stack patches: list length = N (=H*W), each element (B,C,p,p)
    patches_stack = torch.stack(patches_list, dim=0)        # (N, B, C, p, p)
    patches_stack = patches_stack.permute(1, 0, 2, 3, 4)   # (B, N, C, p, p)
    patches_stack = patches_stack.contiguous()

    coords = torch.stack(coords_list, dim=0)               # (N, 2)
    coords = coords.unsqueeze(0).repeat(B, 1, 1)           # (B, N, 2)

    return patches_stack, coords


def interfacial_energy(state: torch.Tensor, axis_pairs=None):
    """
    Simple interfacial energy estimator.
    Input:
        state: (B, C, H, W) where channels include TYPE_A and TYPE_B probabilities
    Output:
        energy: (B,) per-batch scalar energy (mean magnitude of gradients between types)
    """
    if state.dim() != 4:
        raise ValueError("state must be (B,C,H,W)")

    B, C, H, W = state.shape
    # If two or more channels, difference of first two channels
    if C >= 2:
        field = state[:, 0] - state[:, 1]  # (B, H, W)
    else:
        field = state[:, 0]

    # compute gradients (finite differences)
    dx = torch.abs(field[:, :, 1:] - field[:, :, :-1])   # (B, H, W-1)
    dy = torch.abs(field[:, 1:, :] - field[:, :-1, :])   # (B, H-1, W)

    gx = dx.mean(dim=[1,2]) if dx.numel() > 0 else torch.zeros(B, device=state.device)
    gy = dy.mean(dim=[1,2]) if dy.numel() > 0 else torch.zeros(B, device=state.device)

    energy = (gx + gy) * 0.5
    return energy


def motion_penalty(actions: torch.Tensor, kind: str = "l2"):
    """
    Motion penalty for actions.
    Input:
        actions: either
            - (B, A, H, W)  (global)
            - (B, N, A)     (flattened local per-cell actions)
            - (B, A, N)     (some variants)
    Output:
        penalty: (B,) per-batch scalar (mean squared magnitude)
    """
    if not isinstance(actions, torch.Tensor):
        actions = torch.tensor(actions)

    # normalize shape into (B, A, H, W) or (B, N, A)
    if actions.dim() == 4:
        # (B, A, H, W)
        mag = actions.pow(2).mean(dim=[1,2,3])   # per-batch mean squared action
    elif actions.dim() == 3:
        # ambiguous: (B, N, A) or (B, A, N). Detect by size of last dim
        B, x, z = actions.shape
        if z <= 8:
            # (B, N, A)
            mag = actions.pow(2).mean(dim=[1,2])  # (B,)
        else:
            # treat as (B, A, N)
            mag = actions.pow(2).mean(dim=[1,2])
    elif actions.dim() == 2:
        mag = actions.pow(2).mean(dim=1)
    else:
        mag = actions.pow(2).mean().view(1).repeat(actions.shape[0] if actions.dim() > 0 else 1)

    return mag
