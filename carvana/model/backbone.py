from torch import nn
from detectron2.modeling import BACKBONE_REGISTRY, Backbone


@BACKBONE_REGISTRY.register()
class Backbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()

        in_channels = 3
        out_channels = 64
        self.diff_dims = in_channels != out_channels

        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )

        self.bn1 = nn.BatchNorm3d(out_channels)
        for i in self.bn1.parameters():
            i.requires_grad = False

        self.bn2 = nn.BatchNorm3d(out_channels)
        for i in self.bn1.parameters():
            i.requires_grad = False

        self.relu = nn.PReLU()

        self.downsample = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(out_channels),
        )
        for i in self.downsample._modules["1"].parameters():
            i.requires_grad = False

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.diff_dims:
            residual = self.downsample(residual)

        out += residual
        return {"conv3d": out}

    def output_shape(self):
        raise NotImplementedError
