# Train WebTaste on your ratings

Extract this add-on's files directly into your **existing WebTaste project
folder**, beside `webtaste.py` and `data/`. It contains no dataset, model, or
ratings and will not overwrite your data. Close the rating app before training
to take a stable snapshot of your labels.

## Install for Windows / NVIDIA

Use your existing `.venv`. These matched PyTorch 2.11 / torchvision 0.26 packages
use CUDA 12.8 and are listed by the official PyTorch installation documentation:
https://pytorch.org/get-started/previous-versions/ . They require a compatible
NVIDIA driver; a separate CUDA toolkit is not required for these binary wheels.

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

If your environment already has a working matched CUDA torch/torchvision pair,
you can keep it and install only `requirements-ml.txt`. CPU-only installation:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
```

## Train

```powershell
.\.venv\Scripts\python.exe train.py --data data --out models --device cuda
```

First execution downloads the pretrained OpenCLIP checkpoint. Then it embeds
your rated screenshots in batches of 32. Embeddings are computed on the GPU;
the small regressors train on CPU. This is intentional: the expensive image
encoder is frozen, and the scoring models are small.

Default encoder: `ViT-B-32`, pretrained `laion2b_s34b_b79k`, following OpenCLIP:
https://github.com/mlfoundations/open_clip . Whole screenshots are padded to a
square before the model's resize, preserving both horizontal edges. The model
still sees a low-resolution view, so fine typography and small text may be lost.
Training and prediction use the same preprocessing and L2-normalized features.

If CUDA is unavailable use `--device cpu`. If GPU memory is insufficient:

```powershell
.\.venv\Scripts\python.exe train.py --data data --out models --device cuda --batch-size 8
```

Completed embedding batches are cached by image hash and encoder configuration
in `data/embedding_cache/`. Stop with Ctrl+C and rerun to reuse them. Changes to
ratings do not require new embeddings for unchanged images. A missing or corrupt
rated screenshot stops training instead of silently dropping your labels.

## How evaluation works

- Approximately 70% training, 15% validation, 15% final test, by **groups**.
  Actual image counts vary with group sizes.
- Registered original domains, final redirect domains, and identical image
  hashes connect screenshots into groups. Each group stays in one partition.
  Shared templates and near-identical images still need manual review.
- Exact duplicate captures stay inside the same group; reported distinct image
  hashes tell you whether the image count overstates dataset diversity.
- A training-set median baseline is compared with regularized Ridge regression
  and a one-hidden-layer neural network with 32 units.
- Feature scaling and target scaling learn from training only. Ridge strength
  and neural-network early stopping use validation only.
- The winner, including the baseline if it wins, is selected by validation MAE.
  The saved models are then evaluated on the test partition, without refitting.
- Splits and label/image snapshots are saved. Changed ratings or split settings
  require a new output folder. A completed run will not overwrite its model.

The default pilot requires at least 30 rated screenshots and 20 distinct groups;
these are practical guardrails, not a guarantee of meaningful accuracy. Your
672 ratings are sufficient to attempt it, provided enough different sites remain.

## Outputs

| File | Content |
|---|---|
| `models/taste_model.joblib` | Baseline, Ridge, MLP, winner, encoder configuration |
| `models/report.json` | Validation/test MAE, RMSE, Spearman, score distribution, versions, training history |
| `models/report.png` | Rating distribution, held-out predictions, MLP training/validation curves |
| `models/test_predictions.csv` | Your scores versus held-out predictions and image paths |
| `models/validation_predictions.csv` | Selected model's validation predictions |
| `models/splits.json` | Fixed image/group partitions and rating snapshot |

MAE is average absolute error in **0–10 rating points**. Spearman measures whether
the predicted ordering tracks your ordering; it is undefined for constant
predictions and is reported as `null`. Within-one/two-point fractions are also
reported. The 95% bootstrap interval resamples test groups for the fixed selected
model; it does not capture variation from retraining or different dataset splits.

Positive `test_mae_improvement_over_baseline` means lower test MAE than always
predicting your training median. A baseline victory, worse test MAE, or low
ranking correlation is useful evidence that the current model is insufficient.
Do not choose a different winner or repeatedly tune settings using the test
results; use fresh unseen screenshots for the next honest evaluation.

## Predict scores

```powershell
# One screenshot:
.\.venv\Scripts\python.exe predict.py --input screenshot.png

# Directory of images; predictions.csv is sorted from highest score down:
.\.venv\Scripts\python.exe predict.py --input new_screenshots --output predictions.csv

# Captured sites you have not rated, with domains in the CSV:
.\.venv\Scripts\python.exe predict.py --unrated --data data --output unrated_predictions.csv
```

Prediction never changes your ratings or moves screenshots. It marks images
already included in the rated dataset. A baseline winner produces identical
scores and cannot rank designs. Use `--candidate ridge` or `--candidate mlp`
only to inspect alternatives; it does not change the validation-selected winner.

If a CUDA-trained run is moved to a CPU machine, `--device cpu` works: the saved
regressors are CPU models, and CLIP weights are loaded separately. Keep the
Python scripts alongside the model and use matching dependency versions.
Load only trusted `.joblib` models, since this serialization executes Python
pickle data when loaded.

To inspect a future run in another folder:

```powershell
.\.venv\Scripts\python.exe train.py --data data --out models-next --device cuda
.\.venv\Scripts\python.exe predict.py --model models-next/taste_model.joblib --input screenshot.png
```

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_taste_ml
```

These tests use synthetic fixtures to verify grouping, split persistence,
cache reuse, padding, training-only statistics, baseline selection, model
serialization, reports, and prediction. They do not establish accuracy on your
real screenshots. The full CLIP checkpoint requires downloading and running
locally; no real training results are bundled with this add-on.
