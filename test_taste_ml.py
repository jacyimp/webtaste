"""Synthetic-data checks; they do not measure real screenshot quality."""
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
from PIL import Image

from taste_ml import (DEFAULT_ENCODER, RatedImage, assign_groups, embeddings,
                      grouped_mae_interval, make_splits, metrics,
                      square_image, train_models)
import predict
import train


def records(count=100):
    result = [RatedImage(i + 1, f'site{i}.com', '', i % 11, f'hash{i}', Path(f'{i}.png')) for i in range(count)]
    assign_groups(result)
    return result


class FakeEncoder:
    calls = 0

    def __init__(self, config, device):
        self.config = config

    def encode(self, paths):
        FakeEncoder.calls += 1
        return np.array([[float(Path(path).stem), 1.0, 2.0] for path in paths], dtype=np.float32)


class MLTests(unittest.TestCase):
    def test_groups_connect_redirects_and_duplicate_hashes(self):
        items = [
            RatedImage(1, 'a.alpha.co.uk', 'https://beta.com/', 5, 'a', Path('1.png')),
            RatedImage(2, 'www.beta.com', 'https://gamma.com/', 6, 'b', Path('2.png')),
            RatedImage(3, 'gamma.com', '', 5, 'c', Path('3.png')),
            RatedImage(4, 'different.org', '', 4, 'a', Path('4.png')),
            RatedImage(5, 'one.github.io', '', 5, 'd', Path('5.png')),
            RatedImage(6, 'two.github.io', '', 5, 'e', Path('6.png')),
        ]
        assign_groups(items)
        self.assertEqual(len({r.group for r in items[:4]}), 1)
        self.assertNotEqual(items[4].group, items[5].group)

    def test_saved_splits_are_stable_and_do_not_leak_groups(self):
        items = records()
        items[1].sha256 = items[0].sha256
        assign_groups(items)
        with tempfile.TemporaryDirectory() as folder:
            first = make_splits(items, folder)
            second = make_splits(items, folder)
            for name in first:
                np.testing.assert_array_equal(first[name], second[name])
            group_sets = [{items[i].group for i in positions} for positions in first.values()]
            self.assertFalse(group_sets[0] & group_sets[1])
            self.assertFalse(group_sets[0] & group_sets[2])
            self.assertFalse(group_sets[1] & group_sets[2])
            items[0].score = 9
            with self.assertRaisesRegex(ValueError, 'changed'):
                make_splits(items, folder)

    def test_square_padding_preserves_edges(self):
        image = Image.new('RGB', (1440, 900), 'red')
        square = square_image(image)
        self.assertEqual(square.size, (1440, 1440))
        self.assertEqual(square.getpixel((0, 270)), (255, 0, 0))
        self.assertEqual(square.getpixel((1439, 1169)), (255, 0, 0))
        self.assertEqual(square.getpixel((0, 0)), (127, 127, 127))

    def test_embedding_cache_reuse_and_changed_image(self):
        FakeEncoder.calls = 0
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path('1.png'), Path('2.png')]
            first = embeddings(paths, ['digest1', 'digest2'], folder, encoder_factory=FakeEncoder)
            self.assertEqual(FakeEncoder.calls, 1)
            second = embeddings(paths, ['digest1', 'digest2'], folder, encoder_factory=FakeEncoder)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(FakeEncoder.calls, 1)
            embeddings(paths, ['digest1', 'changed'], folder, encoder_factory=FakeEncoder)
            self.assertEqual(FakeEncoder.calls, 2)

    def test_model_selection_does_not_read_test_labels(self):
        rng = np.random.default_rng(3)
        features = rng.normal(size=(100, 8))
        scores = np.clip(5 + features[:, 0] * 2, 0, 10)
        splits = {'train': np.arange(70), 'validation': np.arange(70, 85), 'test': np.arange(85, 100)}
        scores[splits['test']] = np.nan
        models, selected, validation, _ = train_models(features, scores, splits, epochs=8, patience=3)
        self.assertIn(selected, ('baseline', 'ridge', 'mlp'))
        self.assertTrue(all(np.isfinite(value['mae']) for value in validation.values()))
        np.testing.assert_allclose(models['ridge'].scaler.mean_, features[splits['train']].mean(axis=0))

    def test_baseline_can_win_and_correlations_handle_constants(self):
        features = np.zeros((100, 4))
        scores = np.tile([0.0, 10.0], 50)
        splits = {'train': np.arange(60), 'validation': np.arange(60, 80), 'test': np.arange(80, 100)}
        models, selected, _, _ = train_models(features, scores, splits, epochs=4, patience=2)
        self.assertEqual(selected, 'baseline')
        self.assertIsNone(metrics(scores, np.full(100, 5.0))['spearman'])
        self.assertEqual(models['baseline'].predict(features).tolist(), [5.0] * 100)

    def test_group_bootstrap_handles_perfect_predictions(self):
        self.assertEqual(grouped_mae_interval([1, 2, 3], [1, 2, 3], ['a', 'b', 'b']), [0.0, 0.0])

    def test_complete_training_reporting_and_prediction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / 'data'
            data.mkdir()
            (data / 'screenshots').mkdir()
            with sqlite3.connect(data / 'dataset.sqlite3') as db:
                db.execute('CREATE TABLE sites(id INTEGER PRIMARY KEY,domain TEXT,final_url TEXT,score INTEGER,rating_state TEXT,capture_status TEXT)')
                for i in range(1, 101):
                    image = Image.new('RGB', (32, 20), (i, i, i))
                    image.save(data / 'screenshots' / f'{i:07d}.png')
                    db.execute('INSERT INTO sites VALUES(?,?,?,?,?,?)',
                               (i, f'site{i}.com', '', i % 11, 'rated', 'captured'))

            def synthetic_features(paths, *args, **kwargs):
                # Deliberately informative fixtures; this is a functionality test.
                return np.array([[int(p.stem) % 11, int(p.stem) % 3, 0.0, 1.0] for p in paths], dtype=np.float32)

            output = root / 'model'
            with patch('train.embeddings', side_effect=synthetic_features):
                self.assertEqual(train.main(['--data', str(data), '--out', str(output), '--epochs', '12', '--patience', '4']), 0)
            report = json.loads((output / 'report.json').read_text())
            self.assertGreater(report['test_mae_improvement_over_baseline'], 0)
            self.assertTrue((output / 'report.png').is_file())
            self.assertTrue((output / 'test_predictions.csv').is_file())
            self.assertEqual(sum(report['split_counts'].values()), 100)
            bundle = joblib.load(output / 'taste_model.joblib')
            self.assertEqual(bundle['feature_dimension'], 4)
            target = root / 'predictions.csv'
            with patch('predict.embeddings', side_effect=synthetic_features):
                self.assertEqual(predict.main(['--model', str(output / 'taste_model.joblib'), '--input',
                                              str(data / 'screenshots'), '--output', str(target)]), 0)
            with target.open(newline='') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 100)
            self.assertTrue(all(r['seen_in_rated_dataset'] == 'True' for r in rows))
            values = [float(row['predicted_score']) for row in rows]
            self.assertEqual(values, sorted(values, reverse=True))


if __name__ == '__main__':
    unittest.main()
