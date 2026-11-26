"""CSV-backed frame manifests and strict video-level split validation."""
from pathlib import Path

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset


LABELS = {"REAL": 0, "FAKE": 1, "0": 0, "1": 1}


def load_manifest(csv_path, image_column="image_path", label_column="label", video_column="video_id"):
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    frame = pd.read_csv(path)
    required = {image_column, label_column, video_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Manifest is empty: {path}")
    if frame[[image_column, label_column, video_column]].isnull().any().any():
        raise ValueError(f"Manifest has null path, label, or video ID: {path}")
    normalized = frame[label_column].astype(str).str.strip().str.upper()
    unknown = sorted(set(normalized) - set(LABELS))
    if unknown:
        raise ValueError(f"Unsupported labels {unknown}; use REAL/FAKE or 0/1")
    frame = frame.copy()
    frame[label_column] = normalized.map(LABELS).astype("int64")
    frame[image_column] = frame[image_column].astype(str)
    frame[video_column] = frame[video_column].astype(str)
    return frame


def validate_split_manifests(manifests, image_column="image_path", label_column="label", video_column="video_id"):
    """Validate named CSV splits; raises on video overlap or inconsistent video labels."""
    frames = {name: load_manifest(path, image_column, label_column, video_column)
              for name, path in manifests.items() if path is not None}
    owners = {}
    for split, frame in frames.items():
        for video in frame[video_column].unique():
            if video in owners:
                raise ValueError(f"Video leakage: {video!r} occurs in {owners[video]!r} and {split!r}")
            owners[video] = split
        labels_per_video = frame.groupby(video_column)[label_column].nunique()
        inconsistent = labels_per_video[labels_per_video > 1].index.tolist()
        if inconsistent:
            raise ValueError(f"Videos have inconsistent labels in {split}: {inconsistent[:5]}")
    return {name: len(frame) for name, frame in frames.items()}


class FrameDataset(Dataset):
    def __init__(self, manifest, image_size=224, image_column="image_path", label_column="label",
                 video_column="video_id"):
        self.frame = load_manifest(manifest, image_column, label_column, video_column).reset_index(drop=True)
        self.image_column, self.label_column, self.video_column = image_column, label_column, video_column
        self.image_size = int(image_size)
        if self.image_size < 1:
            raise ValueError("image_size must be positive")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row[self.image_column])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)
        return {"image": tensor, "label": torch.tensor(int(row[self.label_column]), dtype=torch.long),
                "video_id": str(row[self.video_column]), "image_path": str(path)}
