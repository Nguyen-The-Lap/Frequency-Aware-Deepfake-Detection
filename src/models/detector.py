import torch
from torch import nn

from .frequency import FrequencyBands, FrequencyCNN


class DeepfakeDetector(nn.Module):
    """Binary detector: RGB EfficientNet-B0, DCT frequency CNN, or fused branches."""
    def __init__(self, kind="rgb", pretrained=True, fusion="concat", num_classes=2, frequency_dim=128,
                 frequency_method="dct"):
        super().__init__()
        if kind not in {"rgb", "frequency", "dual"}:
            raise ValueError("kind must be rgb, frequency, or dual")
        if fusion not in {"concat", "gated"}:
            raise ValueError("fusion must be concat or gated")
        self.kind, self.fusion = kind, fusion
        if kind in {"rgb", "dual"}:
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.rgb = efficientnet_b0(weights=weights)
            rgb_dim = self.rgb.classifier[1].in_features
            self.rgb.classifier = nn.Identity()
            self.rgb_dim = rgb_dim
        if kind in {"frequency", "dual"}:
            self.bands = FrequencyBands(method=frequency_method)
            self.frequency = FrequencyCNN(feature_dim=frequency_dim)
            self.frequency_dim = frequency_dim
        if kind == "rgb":
            combined = self.rgb_dim
        elif kind == "frequency":
            combined = self.frequency_dim
        elif fusion == "concat":
            combined = self.rgb_dim + self.frequency_dim
        else:
            combined = self.rgb_dim
            self.frequency_projection = nn.Linear(self.frequency_dim, self.rgb_dim)
            self.gate = nn.Linear(self.rgb_dim * 2, self.rgb_dim)
        self.classifier = nn.Linear(combined, num_classes)

    def forward(self, x):
        if self.kind == "rgb":
            features = self.rgb(x)
        elif self.kind == "frequency":
            features = self.frequency(self.bands(x))
        else:
            rgb, freq = self.rgb(x), self.frequency(self.bands(x))
            if self.fusion == "concat":
                features = torch.cat((rgb, freq), dim=1)
            else:
                projected = self.frequency_projection(freq)
                gate = torch.sigmoid(self.gate(torch.cat((rgb, projected), dim=1)))
                features = gate * rgb + (1 - gate) * projected
        return self.classifier(features)
