# WebTaste

Can an AI learn your taste in web design?

WebTaste captures website homepages, lets you rate their designs from **0–10**, and trains a model to predict what you like. It also includes a live screen score overlay.

## Setup

Requires Python 3.11 or newer. Open PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## Capture and rate websites

Download website candidates from [Tranco](https://tranco-list.eu/), then capture the first 100 using four workers:

```powershell
python webtaste.py download --top 10000
python webtaste.py capture --limit 100 --workers 4
python webtaste.py rate
```

Run capture again to process the next batch. Completed captures and ratings are saved automatically.

Already have a website list? Import a Tranco ZIP/CSV or a text file of domains instead:

```powershell
python webtaste.py import websites.csv --limit 10000
```

Rate the **visual design of the first screen**: 0 = strongly dislike, 5 = neutral, 10 = strongly like. Skip broken pages, login walls, and screenshots obscured by pop-ups.

| Key | Action |
|---|---|
| `0`–`9` | Assign a score |
| `X` | Assign 10 |
| `S` | Skip |
| `U` or `Ctrl+Z` | Undo |
| `Esc` | Save metadata and close |

Use `python webtaste.py stats` to check progress, or `python webtaste.py rate --review` to change previous ratings.

## Train a model

For an NVIDIA GPU, install the CUDA dependencies and train:

```powershell
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-ml.txt
python train.py --data data --out models --device cuda
```

The first run downloads the pretrained CLIP model. WebTaste compares **Ridge regression**, a **small neural network**, and a **median-rating baseline**. It selects the winner using validation data, then evaluates it on a separate test set. Related websites and identical images stay in the same split.

The saved model and evaluation reports appear in `models/`. See [TRAINING.md](TRAINING.md) for CPU setup and evaluation details.

## Predict scores

Score a screenshot or rank your unrated captures:

```powershell
python predict.py --input screenshot.png
python predict.py --unrated --output unrated_predictions.csv
```

For a transparent, click-through score in the top-right corner of your screen:

```powershell
pythonw screen_taste.py
```

The overlay is Windows-only. Press **Ctrl+Alt+Q** to quit. Add `--interval 0.5` for a shorter update interval or `--crop-top 90` to omit the top 90 pixels. Check `screen_taste.log` if it displays an error.

## Local files

| Folder | Contents |
|---|---|
| `data/screenshots/` | Original captures |
| `data/rated/0/`–`data/rated/10/` | Images sorted by your ratings |
| `data/` | Rating database, metadata, and embedding cache |
| `models/` | Trained models, evaluation reports, and saved splits |

Datasets and models are excluded from Git. Back up the whole `data/` folder after closing the tools.
