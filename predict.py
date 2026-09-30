"""Score a screenshot, an image directory, or WebTaste's unrated captures."""
import argparse
from contextlib import closing
import csv
import hashlib
from pathlib import Path
import sqlite3
import sys

import joblib

from taste_ml import embeddings


def input_images(path, recursive=False):
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f'Input does not exist: {path}')
    extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    entries = path.rglob('*') if recursive else path.iterdir()
    return sorted(p for p in entries if p.is_file() and p.suffix.lower() in extensions)


def unrated_images(data):
    data = Path(data).resolve()
    db_path = data / 'dataset.sqlite3'
    if not db_path.is_file():
        raise ValueError(f'No WebTaste database at {db_path}')
    with closing(sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)) as db:
        rows = db.execute("SELECT id,domain FROM sites WHERE capture_status='captured' AND rating_state='unrated' ORDER BY id").fetchall()
    paths = [data / 'screenshots' / f'{site_id:07d}.png' for site_id, domain in rows]
    if any(not path.is_file() for path in paths):
        raise ValueError('An unrated capture is missing its image. Restore missing originals first.')
    return paths, {str(path): domain for path, (_, domain) in zip(paths, rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/taste_model.joblib'))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path, help='One screenshot or a directory')
    source.add_argument('--unrated', action='store_true', help='Score captured, unrated images in --data')
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--candidate', choices=('selected', 'ridge', 'mlp', 'baseline'), default='selected')
    args = parser.parse_args(argv)
    try:
        if args.batch_size < 1:
            raise ValueError('--batch-size must be positive')
        # Load only model files you produced or trust; joblib uses Python pickles.
        bundle = joblib.load(args.model)
        if bundle.get('format_version') != 1:
            raise ValueError('Unsupported model format')
        if args.unrated:
            paths, domains = unrated_images(args.data)
        else:
            paths, domains = input_images(args.input, args.recursive), {}
        if not paths:
            raise ValueError('No images to score')
        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
        features = embeddings(paths, hashes, args.cache or args.data / 'embedding_cache',
                              bundle['encoder'], args.device, args.batch_size)
        if features.shape[1] != bundle['feature_dimension']:
            raise ValueError('Embedding dimension does not match the trained model')
        selected = bundle['selected'] if args.candidate == 'selected' else args.candidate
        prediction = bundle['models'][selected].predict(features)
        known = set(bundle['dataset_hashes'])
        rows = sorted(zip(paths, hashes, prediction), key=lambda row: float(row[2]), reverse=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['image_path', 'domain', 'predicted_score', 'seen_in_rated_dataset'])
            for path, digest, score in rows:
                writer.writerow([str(path.resolve()), domains.get(str(path), ''), round(float(score), 4), digest in known])
        if selected == 'baseline':
            print('Selected model is the constant median baseline; these scores do not distinguish designs.')
        print(f'Model: {selected}. Scored {len(rows)} images. Highest predictions:')
        for path, digest, score in rows[:10]:
            label = domains.get(str(path), path.name)
            suffix = ' (already in rated dataset)' if digest in known else ''
            print(f'  {float(score):.2f}/10  {label}{suffix}')
        print(f'Saved {args.output.resolve()}')
        return 0
    except KeyboardInterrupt:
        print('\nStopped. Completed embedding batches remain cached.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
