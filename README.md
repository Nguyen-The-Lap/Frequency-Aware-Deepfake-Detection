# Frequency-Aware Deepfake Detection

Research code for binary face-frame classification (`REAL=0`, `FAKE=1`). The project provides a complete RGB EfficientNet-B0 baseline, a differentiable DCT low/mid/high frequency branch, and concatenation or gated two-branch fusion. The baseline trains on frame CSV manifests while validating video-level isolation. Evaluation reports ROC-AUC, accuracy, EER, and confusion matrices at frame and mean-score video level.

## Data and split discipline

No dataset is downloaded or included. Obtain FaceForensics++ (FF++) and Celeb-DF v2 through their official access processes, review each dataset's license and your university storage rules, extract face frames locally, and maintain manifests with columns `image_path,label,video_id`. Paths may be absolute or relative to the process working directory. Labels may be `REAL`/`FAKE` or `0`/`1`. `video_id` must identify the source video consistently across manipulated variants; keep every source video and its derivatives wholly in one split.

Create `data/manifests/ffpp_train.csv`, `ffpp_val.csv`, and `ffpp_test.csv` from the official FF++ video-level split assignments. Do not create random frame-level splits. Keep Celeb-DF v2 in a separate manifest used only for unseen evaluation, never training or model selection. This repository does not ship dataset lists, paths, or performance claims.

## Windows CMD setup and run

Run these from the repository root in **Command Prompt**. Install a PyTorch build appropriate for your CUDA setup from the official PyTorch selector first if needed; CPU PyTorch is sufficient for the smoke tests.

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pytest -q -p no:cacheprovider --basetemp=.pytest-tmp
python scripts\validate_manifests.py --train data\manifests\ffpp_train.csv --val data\manifests\ffpp_val.csv --test data\manifests\ffpp_test.csv --unseen data\manifests\celebdf_v2.csv
python scripts\train.py --config configs\default.yaml
python scripts\evaluate.py --manifest data\manifests\ffpp_test.csv --checkpoint outputs\checkpoints\ffpp_rgb.pt --kind rgb
python scripts\evaluate.py --manifest data\manifests\celebdf_v2.csv --checkpoint outputs\checkpoints\ffpp_rgb.pt --kind rgb
python scripts\evaluate.py --manifest data\manifests\ffpp_test.csv --checkpoint outputs\checkpoints\ffpp_rgb.pt --kind rgb --corruption jpeg --severity 3
```

Training can initialize pretrained ImageNet weights when available; this may fetch weights from torchvision on first use. Pass `--no-pretrained` to avoid fetching and use random initialization. No training is run automatically. Set `model.kind` to `frequency` or `dual` and `model.fusion` to `concat` or `gated` for the corresponding experiments. Test and unseen manifests are deliberately excluded from training and validation.

## Code map

- `configs/`: YAML experiment defaults.
- `src/datasets/`: CSV validation and OpenCV frame dataset.
- `src/models/`: EfficientNet RGB, DCT bands/CNN, and fusion.
- `src/training/`: AdamW, validation scheduler, early stopping, checkpoints.
- `src/evaluation/`: frame/video metrics.
- `scripts/`: manifest checks, training, evaluation.
- `tests/`: synthetic CPU smoke tests; no real faces or benchmark results.
- `outputs/`: local experiment outputs (ignored by git).

## Planned research experiments

Compression, blur, resize, and noise robustness; FFT versus DCT versus SRM front ends; frequency-only, RGB-only, concatenation, and gated fusion ablations. Evaluation supports one corruption at a selected severity and model variants via `--frequency-method`. `configs/experiments.yaml` records a proposed grid. Run each condition on the same split manifests, document severity and seeds, and save predictions/configuration alongside metrics. Official split manifests and actual experiment runs remain to be prepared before reporting results.
