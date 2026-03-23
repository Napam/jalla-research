import torch
from icecream import ic  # noqa: F401


class ConvClassifier(torch.nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()

        layer1_out_chans = 4

        self.conv1_sm = Conv2dFlat(
            in_channels=1, out_channels=layer1_out_chans, kernel_size=3, padding="same", padding_mode="zeros"
        )
        self.conv1_md = Conv2dFlat(
            in_channels=1, out_channels=layer1_out_chans, kernel_size=5, padding="same", padding_mode="zeros"
        )
        self.conv1_lg = Conv2dFlat(
            in_channels=1, out_channels=layer1_out_chans, kernel_size=7, padding="same", padding_mode="zeros"
        )

        self.conv2 = Conv2dFlat(in_channels=layer1_out_chans * 3, out_channels=128, kernel_size=3)

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, n_classes),
        )

        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x: torch.Tensor):
        out1 = self.leaky_relu(self.conv1_sm(x))
        out2 = self.leaky_relu(self.conv1_md(x))
        out3 = self.leaky_relu(self.conv1_lg(x))

        layer1_out = torch.cat([out1, out2, out3], dim=1)
        layer2_out = self.leaky_relu(self.conv2(layer1_out))

        pooled = self.gap(layer2_out).squeeze(-1).squeeze(-1)
        logits = self.classifier(pooled)

        return logits


class Conv2dFlat(torch.nn.Module):
    """Conv2d that stores its weight as a 2D parameter so Muon can optimize it directly.

    Internally reshapes [C_out, C_in*kH*kW] -> [C_out, C_in, kH, kW] in forward().
    The reshape is a zero-copy view, so there's no overhead.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs):
        super().__init__()
        tmp = torch.nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self._stride = tmp.stride
        self._padding = tmp.padding
        self._dilation = tmp.dilation
        self._groups = tmp.groups
        self._weight_shape = tmp.weight.shape
        self.weight = torch.nn.Parameter(tmp.weight.data.view(out_channels, -1))
        self.bias = tmp.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w4d = self.weight.view(self._weight_shape)
        return torch.nn.functional.conv2d(x, w4d, self.bias, self._stride, self._padding, self._dilation, self._groups)
