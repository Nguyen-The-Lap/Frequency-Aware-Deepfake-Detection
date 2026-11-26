import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.manifest import FrameDataset, validate_split_manifests
from src.models import DeepfakeDetector
from src.training import Trainer
from src.utils.reproducibility import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Train a binary deepfake detector.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--kind", choices=["rgb", "frequency", "dual"])
    parser.add_argument("--fusion", choices=["concat", "gated"])
    parser.add_argument("--frequency-method", choices=["dct", "fft", "srm"], default="dct")
    parser.add_argument("--no-pretrained", action="store_true", help="Initialize EfficientNet randomly; do not fetch weights")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data, model_cfg = cfg["data"], cfg["model"]
    cols = (data["image_column"], data["label_column"], data["video_column"])
    counts = validate_split_manifests({"train": data["train_csv"], "val": data["val_csv"],
                                       "test": data["test_csv"]}, *cols)
    print(f"Validated FF++ manifests: {counts}")
    seed_everything(cfg["seed"])
    device = "cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else (
        "cpu" if cfg["device"] == "auto" else cfg["device"])
    common = dict(image_size=cfg["image_size"], image_column=cols[0], label_column=cols[1], video_column=cols[2])
    train_ds, val_ds = FrameDataset(data["train_csv"], **common), FrameDataset(data["val_csv"], **common)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=cfg["num_workers"], pin_memory=device.startswith("cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["num_workers"])
    kind, fusion = args.kind or model_cfg["kind"], args.fusion or model_cfg["fusion"]
    model = DeepfakeDetector(kind=kind, pretrained=model_cfg["pretrained"] and not args.no_pretrained,
                             fusion=fusion, frequency_method=args.frequency_method)
    trainer = Trainer(model, device, cfg["learning_rate"], cfg["weight_decay"], cfg["patience"])
    run = Path(cfg["output_dir"]) / "checkpoints" / f"ffpp_{kind}.pt"
    history = trainer.fit(train_loader, val_loader, cfg["epochs"], run)
    result_path = Path(cfg["output_dir"]) / "logs" / f"ffpp_{kind}_history.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Best checkpoint: {run}")
    # Test and unseen manifests are intentionally not consulted during training.


if __name__ == "__main__":
    main()

