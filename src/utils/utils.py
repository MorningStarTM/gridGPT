import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_survival_from_npy(path,
                           max_steps=8064,
                           smooth_alpha=0.2,
                           title="Survival Steps per Episode",
                           save_path=None,
                           show=True):
    """
    Load survival steps from a .npy (or .npz) file and plot.
    - Accepts 1D arrays (episodes,) or 2D arrays (episodes, ?); if 2D, uses the first column.
    - If given a .npz, uses the first array it finds unless you pass 'key' via path like 'file.npz::arrname'.
    """
    path = str(path)
    key = None
    if "::" in path:  # allow 'file.npz::arrname'
        path, key = path.split("::", 1)

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.suffix.lower() == ".npz":
        dataz = np.load(path)
        if key is None:
            key = next(iter(dataz.files))  # first key
        steps = dataz[key]
    else:
        steps = np.load(path)

    steps = np.asarray(steps, dtype=float)
    if steps.ndim == 2:
        steps = steps[:, 0]  # use first column by default
    elif steps.ndim != 1:
        raise ValueError(f"Expected 1D or 2D array, got shape {steps.shape}")

    # Clean & clip
    steps = np.nan_to_num(steps, nan=0.0, posinf=max_steps, neginf=0.0)
    steps = np.clip(steps, 0, max_steps)

    # EMA smoothing (optional)
    ema = None
    if smooth_alpha and 0 < smooth_alpha <= 1:
        ema = np.empty_like(steps)
        ema[0] = steps[0]
        for i in range(1, len(steps)):
            ema[i] = smooth_alpha * steps[i] + (1 - smooth_alpha) * ema[i - 1]

    # Plot
    plt.figure(figsize=(10, 5))
    x = np.arange(1, len(steps) + 1)
    plt.plot(x, steps, linewidth=1.5, label="Survival steps (raw)")
    if ema is not None:
        plt.plot(x, ema, linestyle="--", linewidth=2.0, label=f"EMA (α={smooth_alpha})")
    plt.axhline(max_steps, linestyle=":", linewidth=1.0, label=f"Max steps = {max_steps}")

    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Steps survived")
    plt.ylim(0, max_steps * 1.05)
    plt.xlim(1, len(steps))
    plt.grid(True, alpha=0.3)
    plt.legend()

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()