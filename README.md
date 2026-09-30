# WebTaste

Build a dataset of website first screens, score them by your personal taste,
and export images into folders `0` through `10` for a future image model.

## Windows setup

Requires Python 3.11 or newer (the regular python.org installer includes Tkinter).
Open PowerShell in this extracted folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m playwright install chromium
```

If you have a different Python version, use `py -3` in the first command.
Using the virtual environment's Python directly avoids activation-policy issues.

## First 100 captures

This download includes **10,000 candidates from the supplied Tranco V349N list**
already imported into `data/dataset.sqlite3`. You do not need to download or
import that list again. No live screenshots or ratings are included.

```powershell
.\.venv\Scripts\python.exe webtaste.py capture --limit 100
.\.venv\Scripts\python.exe webtaste.py rate
```

Capture attempts 100 candidates; the usable screenshot count may be lower.
Run the capture command again to process the next 100 pending candidates.
Ctrl+C stops capture; completed work is retained. The rating app saves each
decision immediately and opens only unrated screenshots by default.

Keep capture settings consistent across the dataset. Defaults are a 1440 × 900
desktop viewport, pixel ratio 1, English locale, UTC timezone, light preference,
a fresh unsigned-in browser context per domain, a three-second settling wait,
and a bounded extra font wait. Only the first screen is captured, without the
browser toolbar. Animations are disabled during the screenshot. These rules
improve consistency but do not make dynamic pages perfectly reproducible.

Cookie banners stay as displayed. Skip screenshots where banners, CAPTCHAs,
login walls, blank pages, or error screens prevent you from judging the design.
HTTP errors and non-HTML responses are recorded as failed captures. Some error
or challenge pages return HTTP 200 and therefore still require manual review.

## Rating

Question: **How much do I like the visual design of this first screen?**

| Input | Action |
|---|---|
| `0`–`9` | Assign that score |
| `X` or the `10` button | Assign 10 |
| `S` | Skip unusable screenshot |
| `U` or `Ctrl+Z` | Undo the last rating or skip, including from previous sessions |
| `Esc` | Export metadata and close |

Anchors: 0 = strongly dislike, 5 = neutral, 10 = strongly like.
Screenshots are shuffled each session, fitted without cropping or stretching,
and URL/title details are hidden unless you enable them. Brands may still be
recognizable in the screenshots. Avoid holding rating keys down.

```powershell
# Rerate images, or revisit skipped captures:
.\.venv\Scripts\python.exe webtaste.py rate --review
.\.venv\Scripts\python.exe webtaste.py rate --skipped

# See progress and rating distribution:
.\.venv\Scripts\python.exe webtaste.py stats

# Export current metadata without opening the GUI:
.\.venv\Scripts\python.exe webtaste.py export
```

SQLite is authoritative. `metadata.csv` is refreshed after capture/import, on
GUI close, or with Export CSV. Its snapshot is not rewritten on every keypress.
Undo and rerating update the corresponding score folders. Original screenshots
are retained, so rated copies use additional disk space. Back up the whole
`data` directory after closing the tools, not just the rating folders.

## Dataset layout

- `data/screenshots/`: original PNGs named by stable local IDs.
- `data/rated/0/` through `data/rated/10/`: images copied into their score folders.
- `data/dataset.sqlite3`: URLs, source rank/list, capture status/settings,
  original/final URL information, title, timestamps, image hashes, scores,
  skip state, and undo history.
- `data/metadata.csv`: readable metadata with paths relative to `data/`.
- `data/sources/`: supplied source archive and its import manifest.

Capture and rate can run in separate terminals. Click Refresh queue to load
new captures without reopening the rating app. Run only one rating window per
dataset; undo history is shared. Do not manually rearrange rated copies.

```powershell
# Repair score folders after an interrupted file operation:
.\.venv\Scripts\python.exe webtaste.py repair

# Retry failed candidates, with at most 100 attempts:
.\.venv\Scripts\python.exe webtaste.py capture --retry-failed --limit 100

# Optional local HTTP/SOCKS proxy for the capture browser:
.\.venv\Scripts\python.exe webtaste.py capture --limit 100 --proxy http://127.0.0.1:8080
```

Capture uses HTTPS; domains without a working HTTPS homepage are logged as
failures. It does not bypass access challenges. Tranco ranks domains, including
infrastructure domains that do not have useful homepages, so filtering is part
of the experiment. Different domains may redirect to the same site: compare
`final_url` and image hashes when curating the training set. Before training,
deduplicate captures and keep related domains/pages together in the same
training, validation, or test split.

## Add more candidates

All commands accept `--data PATH` **before** the subcommand. Use a separate
folder when changing viewport or experimenting with another source:

```powershell
# Fetch the latest ranked list, or pin an archived list:
.\.venv\Scripts\python.exe webtaste.py --data another_dataset download --top 10000
.\.venv\Scripts\python.exe webtaste.py --data another_dataset download --list-id V349N --top 10000

# Import a downloaded Tranco ZIP/CSV, or a plain list of domains/URLs:
.\.venv\Scripts\python.exe webtaste.py import tranco_V349N-1m.csv.zip --limit 20000
.\.venv\Scripts\python.exe webtaste.py --data demo import demo_urls.txt
.\.venv\Scripts\python.exe webtaste.py --data demo capture --limit 10
.\.venv\Scripts\python.exe webtaste.py --data demo rate
```

Imports deduplicate exact domains and preserve existing scores and captures.
The bundled `demo_urls.txt` is hand-picked, not a popularity ranking. Tranco
source: https://tranco-list.eu/ ; methodology: https://tranco-list.eu/methodology .
Review its source attribution terms when redistributing ranking data.

## Linux / macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python webtaste.py capture --limit 100
.venv/bin/python webtaste.py rate
```

Linux needs Tkinter and Chromium's system dependencies (for example,
`sudo apt install python3-tk` and `python -m playwright install --with-deps chromium`).
A desktop/display is required for rating. On macOS use a Python build with Tk support.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

Tests cover import/deduplication, ZIP handling, resume and retry selection,
rating folders, rerating, persistent undo, skipped queues, interrupted-folder
recovery, missing originals, failed copies, metadata export, and the capture
pipeline using a simulated browser. Live website capture and desktop GUI
interaction require a local browser/display and are separate from those tests.

This starter project builds the dataset. Model training comes after the pilot
has usable screenshots and ratings; it is not included in this version.
