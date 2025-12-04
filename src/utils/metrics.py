# # src/utils/metrics.py
# import torch
# import torch.nn.functional as F

# TYPE_A = 0
# TYPE_B = 1
# ADH    = 2
# MORPH  = 3
# CENTER = 4

# def interfacial_energy(state):
#     """
#     Differentiable estimate of interfacial energy:
#     Sum over neighbors of A-B mismatch, weighted by local adhesion.
#     Lower is better (less A|B boundary).
#     """
#     types = F.softmax(state[:, :2], dim=1)  # ensure valid probs
#     a = types[:, TYPE_A:TYPE_A+1]
#     b = types[:, TYPE_B:TYPE_B+1]
#     adhesion = state[:, ADH:ADH+1]

#     # 4-neighborhood kernel
#     k4 = state.new_tensor([[0., 1., 0.],
#                            [1., 0., 1.],
#                            [0., 1., 0.]]).view(1,1,3,3)

#     # neighbor averages
#     a_nb = F.conv2d(a, k4, padding=1) / 4.0
#     b_nb = F.conv2d(b, k4, padding=1) / 4.0

#     mismatch = (a * b_nb) + (b * a_nb)  # high if neighbors differ
#     # weight by adhesion (higher adhesion penalizes heterotypic contacts more)
#     energy = (adhesion * mismatch).mean(dim=(1,2,3))  # per-batch scalar
#     return energy  # (B,)

# def motion_penalty(actions):
#     """
#     Quadratic penalty for velocities to discourage unnecessary motion.
#     """
#     vx = actions[:, 1:2]
#     vy = actions[:, 2:3]
#     pen = (vx * vx + vy * vy).mean(dim=(1,2,3))
#     return pen  # (B,)
# # src/utils/metrics.py  (append at end)

# def extract_local_patches(state, patch_size=5):
#     """
#     Extracts local 5x5x4 patches centered on every cell (excluding center_mask).
#     state: (B,5,H,W)
#     returns: patches (B, N, C', patch_size, patch_size), where N = H*W, C' = 4
#              centers: tensor of shape (B,N,2) with (y,x) coords
#     """
#     B, C, H, W = state.shape
#     assert C >= 4
#     pad = patch_size // 2
#     # remove center_mask channel for observation
#     obs_field = state[:, :4]
#     # unfold extracts all patches
#     patches = torch.nn.functional.unfold(obs_field, kernel_size=patch_size, padding=pad)
#     # shape: (B, 4*patch_size*patch_size, H*W)
#     patches = patches.transpose(1, 2)  # (B, N, 4*ps*ps)
#     patches = patches.view(B, H * W, 4, patch_size, patch_size)

#     # build centers grid (y,x)
#     ys, xs = torch.meshgrid(
#         torch.arange(H, device=state.device),
#         torch.arange(W, device=state.device),
#         indexing="ij"
#     )
#     coords = torch.stack((ys.flatten(), xs.flatten()), dim=-1).unsqueeze(0).repeat(B, 1, 1)
#     return patches, coords

# src/utils/metrics.py
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
    # after stacking: (N, B, C, p, p) -> permute to (B, N, C, p, p)
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
    Notes:
        - This is a generic, stable proxy for interfacial energy: spatial gradient magnitude
          of the difference in type probability fields.
    """
    if state.dim() != 4:
        raise ValueError("state must be (B,C,H,W)")

    # If channels >=2, assume channel 0 corresponds to TYPE_A (compatible with your wrappers).
    # Use difference between channel 0 and 1 if present, else use channel 0 alone.
    B, C, H, W = state.shape
    if C >= 2:
        field = state[:, 0] - state[:, 1]  # (B, H, W)
    else:
        field = state[:, 0]

    # compute gradients (finite differences)
    # pad=False: compute internal diffs, then compute absolute gradients
    dx = torch.abs(field[:, :, 1:] - field[:, :, :-1])   # (B, H, W-1)
    dy = torch.abs(field[:, 1:, :] - field[:, :-1, :])   # (B, H-1, W)

    # average gradient magnitude across space -> energy per batch
    # normalize by number of comparisons to keep scale stable across sizes
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
        # if last dim small (<=8) assume (B, N, A); else try to be robust
        if z <= 8:
            mag = actions.pow(2).mean(dim=[1,2])  # (B,)
        else:
            # treat as (B, A, N)
            mag = actions.pow(2).mean(dim=[1,2])
    elif actions.dim() == 2:
        # (B, A) or (B, N)
        mag = actions.pow(2).mean(dim=1)
    else:
        # fallback: compute global mean and broadcast
        mag = actions.pow(2).mean().view(1).repeat(actions.shape[0] if actions.dim() > 0 else 1)

    return mag
