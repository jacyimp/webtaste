"""Durable capture metadata and recoverable rating-folder exports."""
import csv
import hashlib
import json
import os
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize_domain(value):
    value = value.strip()
    parsed = urlsplit(value if '://' in value else 'https://' + value)
    if parsed.scheme not in ('http', 'https') or parsed.username or parsed.password:
        raise ValueError('Expected an HTTP(S) website domain')
    domain = (parsed.hostname or '').rstrip('.').lower().encode('idna').decode()
    labels = domain.split('.')
    if len(labels) < 2 or any(not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', x) for x in labels):
        raise ValueError('Invalid domain: ' + value)
    if all(x.isdigit() for x in labels):
        raise ValueError('Use website domains, not IP addresses')
    return domain


def read_domains(text):
    """Accept Tranco rank,domain CSV, a domain/URL column, or plain lines."""
    seen = set()
    for fields in csv.reader(text.splitlines()):
        if not fields or fields[0].strip().startswith('#'):
            continue
        fields = [x.strip() for x in fields]
        ranked = len(fields) >= 2 and fields[0].isdigit()
        rank = int(fields[0]) if ranked else None
        value = fields[1] if ranked else fields[0]
        try:
            domain = normalize_domain(value)
        except (ValueError, UnicodeError):
            continue
        if domain not in seen:
            seen.add(domain)
            yield rank, domain


class Dataset:
    def __init__(self, root='data'):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'screenshots').mkdir(exist_ok=True)
        for score in range(11):
            (self.root / 'rated' / str(score)).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'dataset.sqlite3', timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS sites (
                id INTEGER PRIMARY KEY,
                domain TEXT NOT NULL UNIQUE,
                rank INTEGER,
                source TEXT NOT NULL,
                imported_at TEXT NOT NULL,
                capture_status TEXT NOT NULL DEFAULT 'pending',
                captured_at TEXT,
                final_url TEXT,
                title TEXT,
                http_status INTEGER,
                capture_settings TEXT,
                sha256 TEXT,
                error TEXT,
                rating_state TEXT NOT NULL DEFAULT 'unrated',
                score INTEGER CHECK (score BETWEEN 0 AND 10),
                rated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS rating_events (
                id INTEGER PRIMARY KEY,
                site_id INTEGER NOT NULL,
                old_state TEXT NOT NULL,
                old_score INTEGER,
                old_rated_at TEXT,
                created_at TEXT NOT NULL
            );
        ''')

    def close(self):
        self.db.close()

    def row(self, site_id):
        row = self.db.execute('SELECT * FROM sites WHERE id=?', (site_id,)).fetchone()
        if row is None:
            raise ValueError('Unknown screenshot')
        return row

    def image(self, site_id):
        return self.root / 'screenshots' / f'{site_id:07d}.png'

    def rated_image(self, site_id, score):
        return self.root / 'rated' / str(score) / self.image(site_id).name

    def import_sites(self, domains, source):
        before = self.db.total_changes
        with self.db:
            self.db.executemany(
                'INSERT OR IGNORE INTO sites(domain,rank,source,imported_at) VALUES(?,?,?,?)',
                ((domain, rank, source, now()) for rank, domain in domains),
            )
        return self.db.total_changes - before

    def capture_queue(self, retry_failed=False, limit=100):
        statuses = ('pending', 'failed') if retry_failed else ('pending',)
        marks = ','.join('?' for _ in statuses)
        return list(self.db.execute(
            f'SELECT * FROM sites WHERE capture_status IN ({marks}) ORDER BY id LIMIT ?',
            (*statuses, limit),
        ))

    def captured(self, site_id, *, final_url, title, http_status, settings):
        digest = hashlib.sha256(self.image(site_id).read_bytes()).hexdigest()
        with self.db:
            self.db.execute('''UPDATE sites SET capture_status='captured',captured_at=?,
                final_url=?,title=?,http_status=?,capture_settings=?,sha256=?,error=NULL WHERE id=?''',
                (now(), final_url, title, http_status, json.dumps(settings, sort_keys=True), digest, site_id))

    def failed(self, site_id, error):
        with self.db:
            self.db.execute("UPDATE sites SET capture_status='failed',error=? WHERE id=?", (error[:1000], site_id))

    def rating_queue(self, state='unrated'):
        return [r['id'] for r in self.db.execute(
            "SELECT id FROM sites WHERE capture_status='captured' AND rating_state=? ORDER BY id", (state,)
        ) if self.image(r['id']).is_file()]

    def _sync_folder(self, site_id):
        """SQLite is authoritative; recreate only this screenshot's rated copy."""
        row = self.row(site_id)
        keep = None
        if row['rating_state'] == 'rated':
            keep = self.rated_image(site_id, row['score'])
            if not keep.exists():
                temporary = keep.with_suffix('.tmp')
                try:
                    shutil.copyfile(self.image(site_id), temporary)
                    os.replace(temporary, keep)
                finally:
                    temporary.unlink(missing_ok=True)
        for score in range(11):
            candidate = self.rated_image(site_id, score)
            if candidate != keep:
                candidate.unlink(missing_ok=True)

    def reconcile(self):
        """Recover folder exports after an interrupted rating or undo."""
        missing = []
        for row in self.db.execute("SELECT id FROM sites WHERE capture_status='captured'"):
            if self.image(row['id']).is_file():
                self._sync_folder(row['id'])
            else:
                missing.append(row['id'])
        return missing

    def rate(self, site_id, score=None, *, skip=False):
        if not skip and (type(score) is not int or not 0 <= score <= 10):
            raise ValueError('Score must be an integer from 0 to 10')
        row = self.row(site_id)
        if row['capture_status'] != 'captured' or not self.image(site_id).is_file():
            raise ValueError('Screenshot is missing or has not been captured')
        # Prepare the copy before committing a score. A crash is repaired by reconcile().
        if not skip:
            target = self.rated_image(site_id, score)
            temporary = target.with_suffix('.tmp')
            try:
                shutil.copyfile(self.image(site_id), temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        with self.db:
            self.db.execute('''INSERT INTO rating_events
                (site_id,old_state,old_score,old_rated_at,created_at) VALUES(?,?,?,?,?)''',
                (site_id, row['rating_state'], row['score'], row['rated_at'], now()))
            self.db.execute('UPDATE sites SET rating_state=?,score=?,rated_at=? WHERE id=?',
                ('skipped' if skip else 'rated', None if skip else score, now(), site_id))
        self._sync_folder(site_id)

    def undo(self):
        event = self.db.execute('SELECT * FROM rating_events ORDER BY id DESC LIMIT 1').fetchone()
        if event is None:
            return None
        with self.db:
            self.db.execute('UPDATE sites SET rating_state=?,score=?,rated_at=? WHERE id=?',
                (event['old_state'], event['old_score'], event['old_rated_at'], event['site_id']))
            self.db.execute('DELETE FROM rating_events WHERE id=?', (event['id'],))
        self._sync_folder(event['site_id'])
        return event['site_id']

    def stats(self):
        counts = {key: 0 for key in ('pending', 'failed', 'captured', 'rated', 'skipped', 'unrated')}
        histogram = {str(x): 0 for x in range(11)}
        for state, count in self.db.execute('SELECT capture_status,COUNT(*) FROM sites GROUP BY capture_status'):
            counts[state] = count
        for state, count in self.db.execute("SELECT rating_state,COUNT(*) FROM sites WHERE capture_status='captured' GROUP BY rating_state"):
            counts[state] = count
        for score, count in self.db.execute('SELECT score,COUNT(*) FROM sites WHERE score IS NOT NULL GROUP BY score'):
            histogram[str(score)] = count
        total = sum(counts[x] for x in ('pending', 'failed', 'captured'))
        return {'total': total, **counts, 'scores': histogram}

    def export(self):
        target = self.root / 'metadata.csv'
        temporary = target.with_suffix('.tmp')
        cursor = self.db.execute('SELECT * FROM sites ORDER BY id')
        keys = [x[0] for x in cursor.description] + ['screenshot_path', 'rated_path']
        with temporary.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            for row in cursor:
                item = dict(row)
                item['screenshot_path'] = f"screenshots/{row['id']:07d}.png" if row['capture_status'] == 'captured' else ''
                item['rated_path'] = f"rated/{row['score']}/{row['id']:07d}.png" if row['rating_state'] == 'rated' else ''
                writer.writerow(item)
        os.replace(temporary, target)
        return target
