import random
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from torch.utils.data import DataLoader, Dataset


def make_optimizers(model: torch.nn.Module, lr: float, use_muon: bool, muon_lr: float) -> list[torch.optim.Optimizer]:
    if use_muon:
        muon_params = [p for p in model.parameters() if p.ndim == 2]
        adamw_params = [p for p in model.parameters() if p.ndim != 2]
        return [torch.optim.Muon(muon_params, lr=muon_lr, momentum=0.95), torch.optim.AdamW(adamw_params, lr=lr)]
    return [torch.optim.AdamW(model.parameters(), lr=lr)]


@dataclass
class LossTracker:
    raw: list[float] = field(default_factory=list)
    ema_values: list[float] = field(default_factory=list)
    _ema: float | None = field(default=None, init=False, repr=False)
    # ~50-step window
    _alpha: float = field(default=2 / (50 + 1), init=False, repr=False)

    def record(self, loss: float) -> None:
        self.raw.append(loss)
        self._ema = loss if self._ema is None else self._alpha * loss + (1 - self._alpha) * self._ema
        self.ema_values.append(self._ema)


@dataclass
class RunConfig:
    model: torch.nn.Module
    label: str
    use_muon: bool
    raw_color: str
    ema_color: str
    lr: float = 1e-3
    muon_lr: float = 0.02


def _run_validation(
    configs: list[RunConfig],
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    loss_fn: torch.nn.Module,
    device: torch.device | None,
) -> tuple[list[float], list[float]]:
    """Returns (losses, accuracies) per config."""
    n = len(configs)
    total_loss = [0.0] * n
    correct = [0] * n
    total_samples = 0

    with torch.no_grad():
        for x, y in val_loader:
            if device is not None:
                x, y = x.to(device), y.to(device)
            total_samples += y.size(0)
            for i, cfg in enumerate(configs):
                cfg.model.eval()
                logits = cfg.model(x)
                total_loss[i] += loss_fn(logits, y).item() * y.size(0)
                correct[i] += (torch.argmax(logits, dim=1) == y).sum().item()
                cfg.model.train()

    avg_loss = [tl / total_samples for tl in total_loss]
    accuracy = [c / total_samples for c in correct]
    return avg_loss, accuracy


def _show_predictions(
    configs: list[RunConfig],
    val_ds: Dataset,
    device: torch.device | None,
    n_samples: int = 16,
    classes: list[str] | None = None,
) -> None:
    n_models = len(configs)
    ds_len = len(val_ds)  # type: ignore[arg-type]
    n_samples = min(n_samples, ds_len)

    indices = random.sample(range(ds_len), n_samples)
    samples: list[tuple[torch.Tensor, int]] = [val_ds[i] for i in indices]

    images = torch.stack([s[0] for s in samples])
    true_labels = [s[1] for s in samples]

    ncols = n_samples
    nrows = n_models
    fig, axes = plt.subplots(nrows, ncols, figsize=(min(ncols * 1.8, 30), nrows * 2.2))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for row, cfg in enumerate(configs):
        cfg.model.eval()
        with torch.no_grad():
            batch = images.to(device) if device is not None else images
            preds = torch.argmax(cfg.model(batch), dim=1).cpu().tolist()

        for col in range(n_samples):
            ax: Axes = axes[row, col]
            img = images[col].squeeze(0).cpu().numpy()
            ax.imshow(img, cmap="gray")
            ax.axis("off")

            true_lbl = classes[true_labels[col]] if classes else str(true_labels[col])
            pred_lbl = classes[preds[col]] if classes else str(preds[col])
            is_correct = true_labels[col] == preds[col]
            color = "green" if is_correct else "red"
            ax.set_title(pred_lbl, fontsize=7, color=color)

            if row == 0:
                ax.text(
                    0.5,
                    1.25,
                    true_lbl,
                    transform=ax.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="gray",
                )

        # Row label on the left
        axes[row, 0].text(
            -0.3,
            0.5,
            cfg.label,
            transform=axes[row, 0].transAxes,
            ha="right",
            va="center",
            fontsize=9,
            fontweight="bold",
            rotation=90,
        )

    fig.suptitle("Validation Predictions (green=correct, red=wrong, gray=true label)", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0.05, 0, 1, 0.95))
    plt.show()


def compare(
    configs: list[RunConfig],
    ds: Dataset,
    batch_size: int,
    epochs: int = 5,
    device: torch.device | None = None,
    plot_every: int = 50,
    y_lim: tuple[float, float] | None = None,
    val_ds: Dataset | None = None,
    val_interval: float = 0.1,
    classes: list[str] | None = None,
) -> None:
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]] = DataLoader(ds, batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    steps_per_epoch = len(dataloader)

    optimizers_per_run = [make_optimizers(cfg.model, cfg.lr, cfg.use_muon, cfg.muon_lr) for cfg in configs]
    trackers = [LossTracker() for _ in configs]

    has_val = val_ds is not None
    val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None
    val_trackers: list[LossTracker] = []
    val_accuracies: list[list[float]] = []
    val_steps: list[int] = []
    val_step_interval = max(1, int(val_interval * steps_per_epoch))
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
        val_trackers = [LossTracker() for _ in configs]
        val_accuracies = [[] for _ in configs]

    plt.ion()
    fig: Figure
    ax_val: Axes | None = None

    if has_val:
        ax_train: Axes
        fig, (ax_train, ax_val) = plt.subplots(1, 2, figsize=(18, 8))  # pyright: ignore[reportAssignmentType]
        fig.suptitle("Training vs Validation Loss", fontsize=13, fontweight="bold")
    else:
        ax_train_single: Axes
        fig, ax_train_single = plt.subplots(figsize=(12, 10))  # pyright: ignore[reportAssignmentType]
        ax_train = ax_train_single
        fig.suptitle("Training Loss Comparison", fontsize=13, fontweight="bold")

    raw_lines: list[Line2D] = []
    ema_lines: list[Line2D] = []
    for cfg in configs:
        (raw_line,) = ax_train.plot([], [], color=cfg.raw_color, linewidth=0.6, alpha=0.6, zorder=1)
        (ema_line,) = ax_train.plot([], [], color=cfg.ema_color, linewidth=1.8, label=cfg.label, zorder=2)
        raw_lines.append(raw_line)
        ema_lines.append(ema_line)

    ax_train.set_xlabel("Step")
    ax_train.set_ylabel("Loss")
    ax_train.legend(loc="upper right")
    ax_train.grid(True, linestyle="--", alpha=0.4)
    if y_lim is not None:
        ax_train.set_ylim(y_lim)

    val_lines: list[Line2D] = []
    if has_val:
        assert ax_val is not None
        for cfg in configs:
            (val_line,) = ax_val.plot(
                [], [], color=cfg.ema_color, linewidth=2.0, marker="o", markersize=4, label=cfg.label
            )
            val_lines.append(val_line)
        ax_val.set_xlabel("Step")
        ax_val.set_ylabel("Loss")
        ax_val.legend(loc="upper right")
        ax_val.grid(True, linestyle="--", alpha=0.4)
        if y_lim is not None:
            ax_val.set_ylim(y_lim)

    plt.tight_layout()
    plt.show(block=False)

    n_fixed_lines_train = len(ax_train.lines)

    def _redraw(global_step: int) -> None:
        for i, tracker in enumerate(trackers):
            xs = list(range(len(tracker.raw)))
            raw_lines[i].set_data(xs, tracker.raw)
            ema_lines[i].set_data(xs, tracker.ema_values)

        ax_train.relim()
        ax_train.autoscale_view()

        for line in ax_train.lines[n_fixed_lines_train:]:
            line.remove()
        for e in range(1, epochs):
            boundary = e * steps_per_epoch
            if boundary < len(trackers[0].raw):
                ax_train.axvline(x=boundary, color="#cccccc", linewidth=0.8, linestyle=":")

        parts = [f"{configs[i].label}: {trackers[i].ema_values[-1]:.4f}" for i in range(len(configs))]
        ax_train.set_title(f"step {global_step}  EMA  " + "  |  ".join(parts), fontsize=10)

        if has_val and val_trackers and val_steps:
            assert ax_val is not None
            for i, vt in enumerate(val_trackers):
                val_lines[i].set_data(val_steps, vt.raw)
            ax_val.relim()
            ax_val.autoscale_view()
            acc_parts = [
                f"{configs[i].label}: loss={val_trackers[i].raw[-1]:.4f} acc={val_accuracies[i][-1]:.1%}"
                for i in range(len(configs))
                if val_trackers[i].raw
            ]
            ax_val.set_title("  |  ".join(acc_parts), fontsize=9)

        fig.canvas.draw()
        fig.canvas.flush_events()

    global_step = 0
    for epoch in range(epochs):
        totals = [0.0] * len(configs)

        for step, (x, y) in enumerate(dataloader):
            if device is not None:
                x, y = x.to(device), y.to(device)

            for i, cfg in enumerate(configs):
                logits = cfg.model(x)
                loss = loss_fn(logits, y)

                for opt in optimizers_per_run[i]:
                    opt.zero_grad()
                loss.backward()
                for opt in optimizers_per_run[i]:
                    opt.step()

                loss_val = loss.item()
                totals[i] += loss_val
                trackers[i].record(loss_val)

            if global_step % plot_every == 0:
                _redraw(global_step)

            if step % 100 == 0:
                losses_str = "  ".join(f"{configs[i].label}={trackers[i].raw[-1]:.4f}" for i in range(len(configs)))
                print(f"  epoch {epoch + 1}/{epochs}  step {step}  {losses_str}")

            global_step += 1

            if has_val and val_loader is not None and global_step % val_step_interval == 0:
                v_losses, v_accs = _run_validation(configs, val_loader, loss_fn, device)
                val_steps.append(global_step)
                for i in range(len(configs)):
                    val_trackers[i].record(v_losses[i])
                    val_accuracies[i].append(v_accs[i])
                acc_str = "  ".join(f"{configs[i].label}={v_accs[i]:.2%}" for i in range(len(configs)))
                print(f"  val  loss={'  '.join(f'{vl:.4f}' for vl in v_losses)}  acc  {acc_str}")
                _redraw(global_step - 1)

        avgs = "  ".join(f"{configs[i].label}={totals[i] / steps_per_epoch:.4f}" for i in range(len(configs)))
        print(f"epoch {epoch + 1}/{epochs}  avg_loss  {avgs}")

    _redraw(global_step - 1)
    plt.ioff()

    if has_val and val_ds is not None:
        _show_predictions(configs, val_ds, device, n_samples=16, classes=classes)

    plt.show(block=True)
