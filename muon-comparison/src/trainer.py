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


def compare(
    configs: list[RunConfig],
    ds: Dataset,
    batch_size: int,
    epochs: int = 5,
    device: torch.device | None = None,
    plot_every: int = 50,
    y_lim: tuple[float, float] | None = None,
) -> None:
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    steps_per_epoch = len(dataloader)

    optimizers_per_run = [make_optimizers(cfg.model, cfg.lr, cfg.use_muon, cfg.muon_lr) for cfg in configs]
    trackers = [LossTracker() for _ in configs]

    plt.ion()
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(12, 10))  # pyright: ignore[reportAssignmentType]
    fig.suptitle("Training Loss Comparison", fontsize=13, fontweight="bold")

    raw_lines: list[Line2D] = []
    ema_lines: list[Line2D] = []
    for cfg in configs:
        (raw_line,) = ax.plot([], [], color=cfg.raw_color, linewidth=0.6, alpha=0.6)
        (ema_line,) = ax.plot([], [], color=cfg.ema_color, linewidth=1.8, label=cfg.label)
        raw_lines.append(raw_line)
        ema_lines.append(ema_line)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    if y_lim is not None:
        ax.set_ylim(y_lim)

    plt.tight_layout()
    plt.show(block=False)

    n_fixed_lines = len(ax.lines)

    def _redraw(global_step: int) -> None:
        for i, tracker in enumerate(trackers):
            xs = list(range(len(tracker.raw)))
            raw_lines[i].set_data(xs, tracker.raw)
            ema_lines[i].set_data(xs, tracker.ema_values)

        ax.relim()
        ax.autoscale_view()

        for line in ax.lines[n_fixed_lines:]:
            line.remove()
        for e in range(1, epochs):
            boundary = e * steps_per_epoch
            if boundary < len(trackers[0].raw):
                ax.axvline(x=boundary, color="#cccccc", linewidth=0.8, linestyle=":")

        parts = [f"{configs[i].label}: {trackers[i].ema_values[-1]:.4f}" for i in range(len(configs))]
        ax.set_title(f"step {global_step}  EMA  " + "  |  ".join(parts), fontsize=10)
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

        avgs = "  ".join(f"{configs[i].label}={totals[i] / steps_per_epoch:.4f}" for i in range(len(configs)))
        print(f"epoch {epoch + 1}/{epochs}  avg_loss  {avgs}")

    _redraw(global_step - 1)
    plt.ioff()
    plt.show(block=True)
