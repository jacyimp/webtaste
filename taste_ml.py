"""Reusable data, embedding, split, training, and prediction utilities."""
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageOps
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


DEFAULT_ENCODER = {'model_name': 'ViT-B-32', 'pretrained': 'laion2b_s34b_b79k',
                   'image_policy': 'square-pad-gray-127-v1', 'normalize': True}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


@dataclass
class RatedImage:
    id: int
    domain: str
    final_url: str
    score: int
    sha256: str
    path: Path
    group: str = ''


def load_ratings(data):
    """Read a consistent snapshot without modifying the capture/rating database."""
    data = Path(data).resolve()
    db_path = data / 'dataset.sqlite3'
    if not db_path.is_file():
        raise ValueError(f'No dataset database at {db_path}. Set --data to your WebTaste data folder.')
    with closing(sqlite3.connect(db_path.as_uri() + '?mode=ro', uri=True)) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT id,domain,final_url,score FROM sites WHERE rating_state='rated' AND capture_status='captured' ORDER BY id").fetchall()
    result = []
    for row in rows:
        score = row['score']
        if type(score) is not int or not 0 <= score <= 10:
            raise ValueError(f'Invalid score for screenshot {row["id"]}')
        filename = f'{row["id"]:07d}.png'
        path = data / 'screenshots' / filename
        if not path.is_file():
            path = data / 'rated' / str(score) / filename
        if not path.is_file():
            raise ValueError(f'Rated screenshot {row["id"]} is missing. Restore it before training.')
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as exc:
            raise ValueError(f'Cannot read screenshot {path}: {exc}') from exc
        result.append(RatedImage(row['id'], row['domain'], row['final_url'] or '', score,
                                 hashlib.sha256(path.read_bytes()).hexdigest(), path))
    if len(result) < 30:
        raise ValueError(f'Found {len(result)} ratings. This pilot requires at least 30; several hundred is preferable.')
    assign_groups(result)
    return result


def assign_groups(records):
    """Connect original domains, redirects, and exact image duplicates transitively."""
    import tldextract
    # The bundled suffix snapshot avoids network lookups during training.
    extract = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    owners = {}
    for i, item in enumerate(records):
        keys = ['image:' + item.sha256]
        for address in (item.domain, item.final_url):
            if not address:
                continue
            host = urlsplit(address if '://' in address else 'https://' + address).hostname or ''
            host = host.rstrip('.').lower()
            registered = extract(host).top_domain_under_public_suffix or host
            if registered:
                keys.append('domain:' + registered)
        for key in keys:
            if key in owners:
                left, right = root(i), root(owners[key])
                parent[max(left, right)] = min(left, right)
            else:
                owners[key] = i
    members = {}
    for i, item in enumerate(records):
        members.setdefault(root(i), []).append(item.id)
    for i, item in enumerate(records):
        item.group = 'site-' + str(min(members[root(i)]))


def dataset_snapshot(records):
    return [{'id': r.id, 'score': r.score, 'sha256': r.sha256, 'group': r.group} for r in records]


def make_splits(records, output, seed=42, test_fraction=0.15, val_fraction=0.15):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'splits.json'
    snapshot = dataset_snapshot(records)
    config = {'seed': seed, 'test_fraction': test_fraction, 'val_fraction': val_fraction}
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['snapshot'] != snapshot or saved['config'] != config:
            raise ValueError('Ratings, images, grouping, or split settings changed. Use a new --out folder; the saved split will not be overwritten.')
        partitions = saved['partitions']
    else:
        groups = np.array([r.group for r in records])
        if len(set(groups)) < 20:
            raise ValueError('Need at least 20 distinct website/image groups for this pilot evaluation.')
        positions = np.arange(len(records))
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        rest, test = next(splitter.split(positions, groups=groups))
        validation_splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction / (1 - test_fraction), random_state=seed + 1)
        train_local, val_local = next(validation_splitter.split(rest, groups=groups[rest]))
        indexes = {'train': rest[train_local], 'validation': rest[val_local], 'test': test}
        if any(len(index) < 3 for index in indexes.values()):
            raise ValueError('A partition has fewer than three images. Add more distinct sites.')
        partitions = {name: [records[i].id for i in index] for name, index in indexes.items()}
        write_json(path, {'config': config, 'snapshot': snapshot, 'partitions': partitions})
    lookup = {r.id: i for i, r in enumerate(records)}
    if set(sum(partitions.values(), [])) != set(lookup):
        raise ValueError('Saved splits do not cover the dataset')
    indexes = {name: np.array([lookup[item] for item in ids], dtype=int) for name, ids in partitions.items()}
    flattened = np.concatenate(list(indexes.values()))
    if len(flattened) != len(set(flattened)):
        raise ValueError('Saved splits contain repeated images')
    group_sets = {name: {records[i].group for i in ids} for name, ids in indexes.items()}
    if (group_sets['train'] & group_sets['validation'] or group_sets['train'] & group_sets['test']
            or group_sets['validation'] & group_sets['test']):
        raise ValueError('Website/image groups overlap across saved splits')
    return indexes


def square_image(image):
    image = ImageOps.exif_transpose(image).convert('RGB')
    side = max(image.size)
    canvas = Image.new('RGB', (side, side), (127, 127, 127))
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas


class ClipEncoder:
    def __init__(self, config, device='auto'):
        import open_clip
        import torch
        self.torch = torch
        if config['image_policy'] != DEFAULT_ENCODER['image_policy'] or not config['normalize']:
            raise ValueError('Unsupported image preprocessing configuration')
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else 'cpu' if device == 'auto' else device
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable. Install a CUDA build of PyTorch, or use --device cpu.')
        label = torch.cuda.get_device_name(0) if self.device == 'cuda' else 'CPU'
        print(f'Embedding device: {label}', flush=True)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config['model_name'], pretrained=config['pretrained'])
        self.model = self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    def encode(self, paths):
        tensors = []
        for path in paths:
            with Image.open(path) as image:
                tensors.append(self.preprocess(square_image(image)))
        batch = self.torch.stack(tensors).to(self.device)
        with self.torch.inference_mode():
            features = self.model.encode_image(batch).float()
            features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return features.cpu().numpy().astype(np.float32)


def embeddings(paths, hashes, cache, config=None, device='auto', batch_size=32, encoder_factory=ClipEncoder):
    config = dict(config or DEFAULT_ENCODER)
    token = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    cache = Path(cache) / token
    cache.mkdir(parents=True, exist_ok=True)
    write_json(cache / 'encoder.json', config)
    vectors = [None] * len(paths)
    missing = []
    for i, digest in enumerate(hashes):
        target = cache / (digest + '.npy')
        if target.exists():
            try:
                vector = np.load(target, allow_pickle=False)
                if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
                    raise ValueError('Invalid cached embedding')
                vectors[i] = vector.astype(np.float32)
            except (OSError, ValueError):
                missing.append(i)
        else:
            missing.append(i)
    print(f'Embeddings cached: {len(paths) - len(missing)}/{len(paths)}', flush=True)
    if missing:
        encoder = encoder_factory(config, device)
        for start in range(0, len(missing), batch_size):
            positions = missing[start:start + batch_size]
            encoded = np.asarray(encoder.encode([paths[i] for i in positions]), dtype=np.float32)
            if encoded.ndim != 2 or len(encoded) != len(positions) or not np.isfinite(encoded).all():
                raise ValueError('Encoder returned invalid vectors')
            for i, vector in zip(positions, encoded):
                target = cache / (hashes[i] + '.npy')
                temporary = target.with_suffix('.tmp')
                with temporary.open('wb') as file:
                    np.save(file, vector, allow_pickle=False)
                os.replace(temporary, target)
                vectors[i] = vector
            print(f'Embedded {min(start + batch_size, len(missing))}/{len(missing)} new images', flush=True)
    sizes = {vector.shape for vector in vectors}
    if len(sizes) != 1:
        raise ValueError('Cached embeddings have inconsistent sizes. Use a new --cache directory.')
    return np.stack(vectors)


@dataclass
class ScoreModel:
    regressor: object
    scaler: object = None
    target_mean: float = 0.0
    target_scale: float = 1.0

    def predict(self, features):
        transformed = features if self.scaler is None else self.scaler.transform(features)
        prediction = self.regressor.predict(transformed) * self.target_scale + self.target_mean
        return np.clip(prediction, 0, 10)


def metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = np.abs(actual - predicted)
    correlation = None
    if len(actual) >= 3 and np.ptp(actual) > 0 and np.ptp(predicted) > 1e-10:
        value = float(spearmanr(actual, predicted).statistic)
        if np.isfinite(value):
            correlation = value
    return {'mae': float(error.mean()), 'rmse': float(np.sqrt(np.mean((actual - predicted) ** 2))),
            'spearman': correlation, 'within_1_point': float(np.mean(error <= 1)),
            'within_2_points': float(np.mean(error <= 2))}


def train_models(features, scores, splits, seed=42, epochs=300, patience=30):
    train, val = splits['train'], splits['validation']
    x_train, y_train = features[train], scores[train]
    x_val, y_val = features[val], scores[val]
    # Every learned statistic comes from training only.
    scaler = StandardScaler().fit(x_train)
    x_scaled = scaler.transform(x_train)
    candidates = {'baseline': ScoreModel(DummyRegressor(strategy='median').fit(x_train, y_train))}
    validation = {'baseline': metrics(y_val, candidates['baseline'].predict(x_val))}
    ridge_search = []
    best_ridge = None
    for alpha in (10.0, 100.0, 1000.0, 10000.0):
        model = ScoreModel(Ridge(alpha=alpha).fit(x_scaled, y_train), deepcopy(scaler))
        result = metrics(y_val, model.predict(x_val))
        ridge_search.append({'alpha': alpha, **result})
        if best_ridge is None or result['mae'] < best_ridge[0]:
            best_ridge = (result['mae'], model, alpha)
    candidates['ridge'] = best_ridge[1]
    validation['ridge'] = metrics(y_val, best_ridge[1].predict(x_val))
    mean = float(y_train.mean())
    scale = max(float(y_train.std()), 1.0)
    mlp = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', solver='adam',
                       alpha=5.0, batch_size=min(64, len(train)), learning_rate_init=0.001,
                       random_state=seed, shuffle=True, early_stopping=False)
    history = []
    best_mae = float('inf')
    best_epoch = 0
    best_mlp = None
    for epoch in range(1, epochs + 1):
        mlp.partial_fit(x_scaled, (y_train - mean) / scale)
        model = ScoreModel(mlp, scaler, mean, scale)
        train_mae = metrics(y_train, model.predict(x_train))['mae']
        val_mae = metrics(y_val, model.predict(x_val))['mae']
        history.append({'epoch': epoch, 'train_mae': train_mae, 'validation_mae': val_mae})
        if val_mae < best_mae - 0.0001:
            best_mae, best_epoch, best_mlp = val_mae, epoch, deepcopy(model)
        if epoch == 1 or epoch % 25 == 0:
            print(f'MLP epoch {epoch}: train MAE={train_mae:.3f}, validation MAE={val_mae:.3f}', flush=True)
        if epoch - best_epoch >= patience:
            break
    candidates['mlp'] = best_mlp
    validation['mlp'] = metrics(y_val, best_mlp.predict(x_val))
    # The constant baseline can win: no forced claim that the learned model works.
    selected = min(validation, key=lambda name: validation[name]['mae'])
    return candidates, selected, validation, {'ridge_search': ridge_search,
        'ridge_alpha': best_ridge[2], 'mlp_best_epoch': best_epoch, 'mlp_history': history}


def grouped_mae_interval(actual, predicted, groups, seed=42, repetitions=2000):
    unique = sorted(set(groups))
    if len(unique) < 2:
        return None
    error = np.abs(np.asarray(actual) - np.asarray(predicted))
    groups = np.asarray(groups)
    sums = np.array([error[groups == group].sum() for group in unique])
    counts = np.array([np.sum(groups == group) for group in unique])
    samples = np.random.default_rng(seed).integers(0, len(unique), size=(repetitions, len(unique)))
    values = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]
