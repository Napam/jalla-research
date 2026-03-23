import math
import random
from typing import Sequence, Sized

import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


def dataset_visualise(
    ds: Dataset,
    n: int,
    seed: int | None = None,
    classes: Sequence[str] | None = None,
) -> None:
    """Visualise n random samples from a PyTorch Dataset in a grid.

    Each item is expected to be a (image, label) tuple where the image is
    either a PIL Image, a numpy array, or a torch.Tensor with shape
    (C, H, W) or (H, W).  Any other first-element type is displayed with a
    plain ``imshow`` call and matplotlib will do its best.

    Args:
        ds: The PyTorch Dataset to sample from.
        n: Number of items to visualise.  Clamped to ``len(ds)`` when the
            dataset is smaller than the requested count.
        seed: Optional integer seed for the random sampler.  Pass an integer
            to get a reproducible grid; leave as ``None`` for a fresh random
            sample every call.
        classes: Optional sequence of class name strings.  When provided,
            numeric labels are mapped to ``classes[label]`` in the title.
            Falls back to the raw number if the label is out of range.

    Raises:
        TypeError: If ds does not support ``__len__`` or ``__getitem__``.
    """
    if not isinstance(ds, Sized):
        raise TypeError(f"Expected Sized object, got {type(ds).__name__}")

    n = min(n, len(ds))

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), n)

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes_flat = [axes] if n == 1 else list(axes.flat)

    for ax, idx in zip(axes_flat, indices):
        item = ds[idx]

        if isinstance(item, (tuple, list)) and len(item) >= 2:
            image, label = item[0], item[1]
            if classes is not None and isinstance(label, int) and 0 <= label < len(classes):
                title = classes[label]
            else:
                title = str(label)
        else:
            image = item
            title = f"[{idx}]"

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu()
            if image.ndim == 3:
                image = image.permute(1, 2, 0)
                if image.shape[2] == 1:
                    image = image.squeeze(2)
            image = image.numpy()

        ax.imshow(image, cmap="viridis" if getattr(image, "ndim", 3) == 2 else None)
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(f"Dataset sample  (n={n})", fontsize=10)
    fig.tight_layout()
    plt.show()
