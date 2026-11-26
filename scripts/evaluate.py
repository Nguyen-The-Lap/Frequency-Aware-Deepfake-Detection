import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.manifest import FrameDataset
from src.evaluation.metrics import aggregate_video_predictions, evaluate_predictions
from src.evaluation.corruptions import corrupt_batch
from src.models import DeepfakeDetector


def main():
    parser = argparse.ArgumentParser(description="Frame and video level evaluation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--kind", choices=["rgb", "frequency", "dual"], default="rgb")
    parser.add_argument("--fusion", choices=["concat", "gated"], default="concat")
    parser.add_argument("--frequency-method", choices=["dct", "fft", "srm"], default="dct")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--corruption", choices=["jpeg", "blur", "resize", "noise"])
    parser.add_argument("--severity", type=int, default=1)
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    ds = FrameDataset(args.manifest, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
    model = DeepfakeDetector(args.kind, pretrained=False, fusion=args.fusion,
                             frequency_method=args.frequency_method).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    model.eval()
    labels, scores, videos = [], [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            if args.corruption:
                images = corrupt_batch(images, args.corruption, args.severity)
            logits = model(images)
            scores.extend(logits.softmax(1)[:, 1].cpu().numpy())
            labels.extend(batch["label"].numpy())
            videos.extend(batch["video_id"])
    print("Frame-level:", evaluate_predictions(labels, scores))
    video = aggregate_video_predictions(videos, labels, scores)
    print("Video-level:", evaluate_predictions(video.label, video.score))


if __name__ == "__main__":
    main()

