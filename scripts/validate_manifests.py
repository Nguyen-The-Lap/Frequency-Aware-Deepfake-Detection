import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.manifest import validate_split_manifests


def main():
    parser = argparse.ArgumentParser(description="Check CSV manifests for video leakage.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--unseen", help="Optional Celeb-DF v2 manifest")
    args = parser.parse_args()
    counts = validate_split_manifests({"train": args.train, "val": args.val, "test": args.test,
                                       "unseen": args.unseen})
    for split, count in counts.items():
        print(f"{split}: {count} frames")
    print("Video-level isolation verified.")


if __name__ == "__main__":
    main()

