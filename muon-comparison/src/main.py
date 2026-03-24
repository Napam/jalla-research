import model
import torch
import torchvision
import trainer
import utils  # noqa: F401


def main() -> None:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    fmnist = torchvision.datasets.FashionMNIST("../data", download=True, transform=torchvision.transforms.ToTensor())
    n_classes = len(fmnist.classes)

    ds_train, ds_val = torch.utils.data.random_split(fmnist, [0.8, 0.2])

    print(f"Train: {len(ds_train)} samples, Val: {len(ds_val)} samples")

    model_muon = model.ConvClassifier(n_classes).to(device)
    model_adamw = model.ConvClassifier(n_classes).to(device)
    model_adamw_high_lr = model.ConvClassifier(n_classes).to(device)

    trainer.compare(
        configs=[
            trainer.RunConfig(
                model=model_muon,
                label="Muon+AdamW",
                use_muon=True,
                raw_color="#a0c4ff",
                ema_color="#0066cc",
                lr=0.001,
                muon_lr=0.02,
            ),
            trainer.RunConfig(
                model=model_adamw,
                label="AdamW only",
                use_muon=False,
                raw_color="#ffb3b3",
                ema_color="#cc0000",
                lr=0.001,
            ),
            trainer.RunConfig(
                model=model_adamw_high_lr,
                label="AdamW only, high LR",
                use_muon=False,
                raw_color="lightgreen",
                ema_color="green",
                lr=0.02,
            ),
        ],
        ds=ds_train,
        val_ds=ds_val,
        batch_size=16,
        device=device,
        epochs=1,
        y_lim=(0, 2.5),
        classes=fmnist.classes,
    )


if __name__ == "__main__":
    main()
