"""Run with: python -m unittest -v"""
import asyncio
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from capture import capture_sites
from dataset import Dataset, read_domains
from webtaste import decode_list, main


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.data = Dataset(self.temporary.name)
        self.data.import_sites([(1, 'example.com'), (2, 'example.org')], 'test')

    def tearDown(self):
        self.data.close()
        self.temporary.cleanup()

    def captured(self, site_id=1):
        Image.new('RGB', (1440, 900), '#722f37').save(self.data.image(site_id))
        self.data.captured(site_id, final_url='https://example.com/', title='Test',
                           http_status=200, settings={'width': 1440, 'height': 900})

    def test_import_normalizes_and_deduplicates(self):
        text = 'rank,domain\n1,EXAMPLE.com\n2,example.com\n3,https://www.python.org/path\ninvalid\n'
        self.assertEqual(list(read_domains(text)), [(1, 'example.com'), (3, 'www.python.org')])
        self.assertEqual(self.data.import_sites([(1, 'example.com')], 'new'), 0)

    def test_resume_excludes_success_and_retries_failures(self):
        self.captured()
        self.data.failed(2, 'Timeout')
        self.assertEqual(self.data.capture_queue(), [])
        self.assertEqual([r['id'] for r in self.data.capture_queue(True)], [2])

    def test_rating_rerating_and_persistent_undo(self):
        self.captured()
        self.data.rate(1, 10)
        self.assertTrue(self.data.rated_image(1, 10).is_file())
        self.data.rate(1, 0)
        self.assertFalse(self.data.rated_image(1, 10).exists())
        self.assertTrue(self.data.rated_image(1, 0).is_file())
        self.data.close()
        self.data = Dataset(self.temporary.name)
        self.assertEqual(self.data.undo(), 1)
        self.assertEqual(self.data.row(1)['score'], 10)
        self.assertTrue(self.data.rated_image(1, 10).is_file())
        self.assertFalse(self.data.rated_image(1, 0).exists())
        self.data.undo()
        self.assertEqual(self.data.rating_queue(), [1])
        self.assertTrue(self.data.image(1).is_file())
        self.assertIsNone(self.data.undo())

    def test_skip_and_revisit(self):
        self.captured()
        self.data.rate(1, skip=True)
        self.assertEqual(self.data.rating_queue(), [])
        self.assertEqual(self.data.rating_queue('skipped'), [1])
        self.data.rate(1, 6)
        self.assertEqual(self.data.rating_queue('rated'), [1])
        self.data.undo()
        self.assertEqual(self.data.rating_queue('skipped'), [1])
        self.assertFalse(self.data.rated_image(1, 6).exists())

    def test_folder_recovery_after_interruption(self):
        self.captured()
        self.data.rate(1, 8)
        self.data.rated_image(1, 8).unlink()
        self.data.rated_image(1, 2).write_bytes(b'orphaned copy')
        self.assertEqual(self.data.reconcile(), [])
        self.assertTrue(self.data.rated_image(1, 8).is_file())
        self.assertFalse(self.data.rated_image(1, 2).exists())

    def test_missing_original_is_reported_not_silently_rated(self):
        self.captured()
        self.data.image(1).unlink()
        self.assertEqual(self.data.reconcile(), [1])
        self.assertEqual(self.data.rating_queue(), [])
        with self.assertRaises(ValueError):
            self.data.rate(1, 5)

    def test_invalid_scores_do_not_mutate(self):
        self.captured()
        for score in [-1, 11, 5.5, True, None]:
            with self.assertRaises(ValueError):
                self.data.rate(1, score)
        self.assertEqual(self.data.row(1)['rating_state'], 'unrated')

    def test_failed_copy_does_not_save_a_rating(self):
        self.captured()
        with patch('dataset.shutil.copyfile', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.data.rate(1, 5)
        self.assertEqual(self.data.row(1)['rating_state'], 'unrated')
        self.assertIsNone(self.data.undo())

    def test_export_and_histogram(self):
        self.captured()
        self.data.rate(1, 7)
        with self.data.export().open(encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(rows[0]['rated_path'], 'rated/7/0000001.png')
        self.assertEqual(rows[0]['score'], '7')
        self.assertEqual(len(rows[0]['sha256']), 64)
        self.assertEqual(self.data.stats()['scores']['7'], 1)
        self.assertEqual(self.data.stats()['pending'], 1)

    def test_csv_and_zip_import_cli(self):
        archive = Path(self.temporary.name) / 'tranco_TEST-1m.csv.zip'
        with zipfile.ZipFile(archive, 'w') as file:
            file.writestr('top-1m.csv', '1,python.org\n2,rust-lang.org\n')
        self.assertIn('python.org', decode_list(archive.read_bytes()))
        self.assertEqual(main(['--data', self.temporary.name, 'import', str(archive), '--limit', '1']), 0)
        self.assertEqual(self.data.stats()['total'], 3)
        row = self.data.db.execute("SELECT * FROM sites WHERE domain='python.org'").fetchone()
        self.assertEqual(row['source'], 'tranco:TEST')


class FakePage:
    def __init__(self):
        self.url = ''

    def set_default_timeout(self, timeout):
        pass

    def on(self, event, callback):
        pass

    async def goto(self, url, **kwargs):
        self.url = url + '/'
        return SimpleNamespace(status=403 if 'blocked' in url else 200,
                               headers={'content-type': 'text/html; charset=utf-8'})

    async def wait_for_timeout(self, milliseconds):
        pass

    async def evaluate(self, script):
        pass

    async def title(self):
        return 'Fixture homepage'

    async def screenshot(self, path, **kwargs):
        Image.new('RGB', (1440, 900), 'white').save(path)


class FakeContext:
    async def new_page(self):
        return FakePage()

    async def close(self):
        pass


class FakeBrowser:
    version = 'fixture'

    async def new_context(self, **kwargs):
        return FakeContext()

    async def close(self):
        pass


class FakePlaywright:
    async def __aenter__(self):
        async def launch(**kwargs):
            return FakeBrowser()
        return SimpleNamespace(chromium=SimpleNamespace(launch=launch))

    async def __aexit__(self, *args):
        pass


class CaptureTests(unittest.TestCase):
    def test_capture_success_failure_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Dataset(folder)
            try:
                data.import_sites([(1, 'example.com'), (2, 'blocked.example')], 'test')
                args = SimpleNamespace(retry_failed=False, limit=100, width=1440, height=900,
                                       settle=0, timeout=30, headed=False, proxy=None, workers=2, delay=0)
                with patch('playwright.async_api.async_playwright', return_value=FakePlaywright()):
                    asyncio.run(capture_sites(data, args))
                    asyncio.run(capture_sites(data, args))
                self.assertEqual(data.stats()['captured'], 1)
                self.assertEqual(data.stats()['failed'], 1)
                self.assertEqual(data.row(2)['error'], 'HTTP 403')
                self.assertEqual(data.rating_queue(), [1])
                self.assertEqual(json.loads(data.row(1)['capture_settings'])['viewport']['width'], 1440)
                with Image.open(data.image(1)) as image:
                    self.assertEqual(image.size, (1440, 900))
                self.assertTrue((data.root / 'metadata.csv').exists())
            finally:
                data.close()


if __name__ == '__main__':
    unittest.main()
