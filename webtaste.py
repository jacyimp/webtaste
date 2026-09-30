"""WebTaste CLI: download -> capture -> rate -> export."""
import argparse
import asyncio
import hashlib
import io
import itertools
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from dataset import Dataset, now, read_domains


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('Must be greater than zero')
    return number


def nonnegative(value):
    number = float(value)
    if not 0 <= number < float('inf'):
        raise argparse.ArgumentTypeError('Must be a finite nonnegative number')
    return number


def request_bytes(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'WebTasteDataset/1.0'})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read(50 * 1024 * 1024 + 1)
    if len(payload) > 50 * 1024 * 1024:
        raise ValueError('Download exceeds 50 MiB')
    return payload


def decode_list(payload):
    if payload.startswith(b'PK'):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = [name for name in archive.namelist() if name.endswith('.csv')]
            if not names:
                raise ValueError('ZIP contains no CSV file')
            entry = archive.getinfo(names[0])
            if entry.file_size > 100 * 1024 * 1024:
                raise ValueError('Uncompressed list exceeds 100 MiB')
            payload = archive.read(names[0])
    return payload.decode('utf-8-sig')


def download(dataset, args):
    list_id = args.list_id or request_bytes('https://tranco-list.eu/top-1m-id').decode().strip()
    if not re.fullmatch(r'[A-Za-z0-9]+', list_id):
        raise ValueError('Invalid Tranco list ID')
    url = f'https://tranco-list.eu/download/{list_id}/1000000/'
    print(f'Downloading Tranco list {list_id}...', flush=True)
    payload = request_bytes(url)
    text = decode_list(payload)
    domains = list(itertools.islice(read_domains(text), args.top))
    if not domains:
        raise ValueError('The downloaded response did not contain website domains')
    sources = dataset.root / 'sources'
    sources.mkdir(exist_ok=True)
    raw_name = f'tranco-{list_id}.zip' if payload.startswith(b'PK') else f'tranco-{list_id}.csv'
    (sources / raw_name).write_bytes(payload)
    manifest = {'provider': 'Tranco', 'list_id': list_id, 'download_url': url,
                'downloaded_at': now(), 'sha256': hashlib.sha256(payload).hexdigest(),
                'top_requested': args.top, 'candidates_selected': len(domains)}
    (sources / f'tranco-{list_id}-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    count = dataset.import_sites(domains, f'tranco:{list_id}')
    print(f'Added {count} new domains ({len(domains)} selected).')


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data', default='data', help='Dataset folder; default: ./data')
    commands = cli.add_subparsers(dest='command', required=True)
    fetch = commands.add_parser('download', help='Import a fixed Tranco popularity list')
    fetch.add_argument('--top', type=positive, default=10000)
    fetch.add_argument('--list-id', help='Use a specific archived list instead of the latest')
    imp = commands.add_parser('import', help='Import rank,domain CSV, ZIP, or a plain URL/domain list')
    imp.add_argument('file', type=Path)
    imp.add_argument('--limit', type=positive, default=10000)
    cap = commands.add_parser('capture', help='Capture pending homepages; rerun to resume')
    cap.add_argument('--limit', type=positive, default=100, help='Maximum attempts in this run')
    cap.add_argument('--workers', type=positive, default=2)
    cap.add_argument('--width', type=positive, default=1440)
    cap.add_argument('--height', type=positive, default=900)
    cap.add_argument('--timeout', type=positive, default=30)
    cap.add_argument('--settle', type=nonnegative, default=3)
    cap.add_argument('--delay', type=nonnegative, default=1, help='Pause per worker between sites')
    cap.add_argument('--retry-failed', action='store_true')
    cap.add_argument('--headed', action='store_true', help='Show the capture browser')
    cap.add_argument('--proxy', help='Browser proxy, e.g. http://127.0.0.1:8080')
    rating = commands.add_parser('rate', help='Open the desktop rating app')
    mode = rating.add_mutually_exclusive_group()
    mode.add_argument('--review', action='store_true', help='Rerate previously rated screenshots')
    mode.add_argument('--skipped', action='store_true', help='Revisit skipped screenshots')
    rating.add_argument('--ordered', action='store_true', help='Keep capture order instead of shuffling')
    commands.add_parser('stats', help='Show capture progress and score distribution')
    commands.add_parser('export', help='Export metadata.csv from SQLite')
    commands.add_parser('repair', help='Rebuild rating folders from SQLite after an interruption')
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    dataset = Dataset(args.data)
    try:
        if args.command == 'download':
            download(dataset, args)
            dataset.export()
        elif args.command == 'import':
            payload = args.file.read_bytes()
            text = decode_list(payload)
            domains = list(itertools.islice(read_domains(text), args.limit))
            if not domains:
                raise ValueError('No valid domains found in that file')
            match = re.search(r'tranco[_-]([A-Za-z0-9]+)', args.file.name)
            source = f'tranco:{match[1]}' if match else f'import:{args.file.name}'
            sources = dataset.root / 'sources'
            sources.mkdir(exist_ok=True)
            digest = hashlib.sha256(payload).hexdigest()
            (sources / (digest[:12] + '-' + args.file.name)).write_bytes(payload)
            manifest = {'source': source, 'original_filename': args.file.name,
                        'imported_at': now(), 'sha256': digest,
                        'limit_requested': args.limit, 'candidates_selected': len(domains)}
            (sources / (digest[:12] + '-manifest.json')).write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            count = dataset.import_sites(domains, source)
            print(f'Added {count} new domains.')
            dataset.export()
        elif args.command == 'capture':
            from capture import capture_sites
            asyncio.run(capture_sites(dataset, args))
        elif args.command == 'rate':
            from rate import RatingApp
            missing = dataset.reconcile()
            if missing:
                print(f'Warning: {len(missing)} original screenshots are missing. Restore them from your backup.')
            state = 'rated' if args.review else 'skipped' if args.skipped else 'unrated'
            RatingApp(dataset, state=state, shuffle=not args.ordered).run()
        elif args.command == 'stats':
            print(json.dumps(dataset.stats(), indent=2))
        elif args.command == 'export':
            print(dataset.export())
        elif args.command == 'repair':
            missing = dataset.reconcile()
            print(f'Rating folders synchronized; {len(missing)} missing original screenshots.')
            dataset.export()
        return 0
    except KeyboardInterrupt:
        dataset.export()
        print('\nStopped. Rerun the same command to resume.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        if args.command == 'download':
            print('You can download the CSV/ZIP from https://tranco-list.eu/ and use the import command.', file=sys.stderr)
        if args.command == 'capture':
            print('If Chromium is missing, run: python -m playwright install chromium', file=sys.stderr)
        return 1
    finally:
        dataset.close()


if __name__ == '__main__':
    raise SystemExit(main())
