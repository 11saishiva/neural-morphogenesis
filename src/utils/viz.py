# src/utils/viz.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from PIL import Image

def state_to_rgb(state):
    """
    Convert a single simulation state (5-channel tensor) into an RGB image.
    Colors:
      - Type A: red intensity
      - Type B: blue intensity
      - Adhesion: green overlay
    """
    with torch.no_grad():
        state = state.detach().cpu()
        a = state[0].numpy()
        b = state[1].numpy()
        adh = state[2].numpy()
        img = np.stack([a, adh, b], axis=-1)
        img = np.clip(img / img.max(), 0, 1)
        return img

def save_gif(frames, path, fps=10):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pil_frames = [Image.fromarray((f * 255).astype(np.uint8)) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )

def visualize_sequence(states, out_path="visuals/sorting.gif", every=1):
    """
    states: list of tensors (B,5,H,W) — uses first batch element for visualization.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    frames = []
    for i, s in enumerate(states):
        if i % every != 0:
            continue
        img = state_to_rgb(s[0])
        frames.append(img)
    save_gif(frames, out_path)
    print(f"[viz] Saved GIF: {out_path}")
