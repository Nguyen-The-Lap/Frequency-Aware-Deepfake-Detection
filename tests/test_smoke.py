import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.datasets.manifest import FrameDataset, validate_split_manifests
from src.evaluation.metrics import aggregate_video_predictions, evaluate_predictions
from src.models.detector import DeepfakeDetector
from src.models.frequency import FrequencyBands


def _manifest(tmp_path, name, video, label):
    image = tmp_path / f"{name}.jpg"
    assert cv2.imwrite(str(image), np.full((24, 24, 3), 127, np.uint8))
    csv = tmp_path / f"{name}.csv"
    pd.DataFrame([{"image_path": str(image), "label": label, "video_id": video}]).to_csv(csv, index=False)
    return csv


def test_manifest_loader_and_frame_dataset(tmp_path):
    csv = _manifest(tmp_path, "one", "vid1", "REAL")
    ds = FrameDataset(csv, image_size=16)
    sample = ds[0]
    assert sample["image"].shape == (3, 16, 16)
    assert sample["label"].item() == 0
    assert sample["video_id"] == "vid1"


def test_video_leakage_rejected(tmp_path):
    a = _manifest(tmp_path, "a", "same-video", "REAL")
    b = _manifest(tmp_path, "b", "same-video", "FAKE")
    with pytest.raises(ValueError, match="Video leakage"):
        validate_split_manifests({"train": a, "test": b})


def test_frequency_branch_cpu_forward_and_gradient():
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    bands = FrequencyBands()(x)
    assert bands.shape == (2, 9, 32, 32)
    bands.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    model = DeepfakeDetector("frequency", pretrained=False)
    logits = model(x.detach())
    assert logits.shape == (2, 2)


@pytest.mark.parametrize("method", ["dct", "fft", "srm"])
def test_frequency_ablation_frontends(method):
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    result = FrequencyBands(method=method)(x)
    assert result.shape == (2, 9, 32, 32)
    result.mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_metrics_frame_and_video_level():
    result = evaluate_predictions([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert result["accuracy"] == 1.0 and result["roc_auc"] == 1.0
    videos = aggregate_video_predictions(["a", "a", "b", "b"], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert len(videos) == 2
    assert evaluate_predictions(videos.label, videos.score)["accuracy"] == 1.0

