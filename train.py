"""Train and evaluate WebTaste on local rated screenshots."""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np

from taste_ml import (DEFAULT_ENCODER, embeddings, grouped_mae_interval,
                      load_ratings, make_splits, metrics, timestamp,
                      train_models, write_json)


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Must be positive')
    return number


def fraction(value):
    number = float(value)
    if not 0 < number < 0.5:
        raise argparse.ArgumentTypeError('Must be between 0 and 0.5')
    return number


def versions():
    result = {}
    for package in ('numpy', 'scikit-learn', 'torch', 'torchvision', 'open_clip_torch', 'tldextract'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def plot_report(path, scores, test_actual, test_predicted, training):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].bar(np.arange(11), np.bincount(scores.astype(int), minlength=11), color='#722f37')
    axes[0].set(title='Your ratings', xlabel='Score', ylabel='Images', xticks=range(11))
    axes[1].scatter(test_actual, test_predicted, s=25, alpha=0.55, color='#722f37')
    axes[1].plot([0, 10], [0, 10], color='#777777', linestyle='--')
    axes[1].set(title='Selected model on held-out test', xlabel='Your score', ylabel='Predicted score',
                xlim=(-0.2, 10.2), ylim=(-0.2, 10.2))
    history = training['mlp_history']
    axes[2].plot([r['epoch'] for r in history], [r['train_mae'] for r in history], label='Training')
    axes[2].plot([r['epoch'] for r in history], [r['validation_mae'] for r in history], label='Validation', color='#722f37')
    axes[2].axvline(training['mlp_best_epoch'], linestyle=':', color='#777777')
    axes[2].set(title='Small neural network', xlabel='Epoch', ylabel='Mean absolute error')
    axes[2].legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data'))
    parser.add_argument('--out', type=Path, default=Path('models'))
    parser.add_argument('--cache', type=Path, help='Default: DATA/embedding_cache')
    parser.add_argument('--device', choices=('auto', 'cuda', 'cpu'), default='auto')
    parser.add_argument('--batch-size', type=positive, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-fraction', type=fraction, default=0.15)
    parser.add_argument('--val-fraction', type=fraction, default=0.15)
    parser.add_argument('--epochs', type=positive, default=300)
    parser.add_argument('--patience', type=positive, default=30)
    parser.add_argument('--encoder', default=DEFAULT_ENCODER['model_name'])
    parser.add_argument('--pretrained', default=DEFAULT_ENCODER['pretrained'])
    parser.add_argument('--embeddings-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.test_fraction + args.val_fraction >= 0.7:
            raise ValueError('Reserve at least 30% for training')
        args.out.mkdir(parents=True, exist_ok=True)
        if (args.out / 'taste_model.joblib').exists():
            raise ValueError('This run already has a model. Use a new --out folder for another experiment.')
        records = load_ratings(args.data)
        splits = make_splits(records, args.out, args.seed, args.test_fraction, args.val_fraction)
        unique = len({r.sha256 for r in records})
        groups = len({r.group for r in records})
        print(f'Rated images: {len(records)} | distinct image hashes: {unique} | website/image groups: {groups}')
        for name, indices in splits.items():
            print(f'{name}: {len(indices)} images, {len({records[i].group for i in indices})} groups')
        scores = np.array([r.score for r in records], dtype=float)
        print('Score distribution:', dict(enumerate(np.bincount(scores.astype(int), minlength=11).tolist())))
        encoder = {**DEFAULT_ENCODER, 'model_name': args.encoder, 'pretrained': args.pretrained}
        cache = args.cache or args.data / 'embedding_cache'
        features = embeddings([r.path for r in records], [r.sha256 for r in records], cache,
                              encoder, args.device, args.batch_size)
        if args.embeddings_only:
            print('Embeddings cached. Rerun without --embeddings-only to train.')
            return 0
        if np.ptp(scores[splits['train']]) == 0:
            raise ValueError('All training ratings are identical. Add varied ratings before fitting a taste model.')
        candidates, selected, validation, training = train_models(
            features, scores, splits, args.seed, args.epochs, args.patience)
        print('\nValidation results (used for model selection):')
        for name, result in validation.items():
            print(f"  {name:8s} MAE={result['mae']:.3f}")
        print(f'Selected by validation MAE: {selected}')
        # Test labels enter evaluation only after hyperparameters and winner are frozen.
        test = splits['test']
        test_predictions = {name: model.predict(features[test]) for name, model in candidates.items()}
        test_metrics = {name: metrics(scores[test], prediction) for name, prediction in test_predictions.items()}
        interval = grouped_mae_interval(scores[test], test_predictions[selected],
                                        [records[i].group for i in test], args.seed)
        report = {
            'created_at': timestamp(), 'rated_images': len(records), 'distinct_images': unique,
            'website_image_groups': groups, 'score_histogram': np.bincount(scores.astype(int), minlength=11).tolist(),
            'split_counts': {name: len(indices) for name, indices in splits.items()},
            'encoder': encoder, 'selected_model': selected, 'validation': validation,
            'test': test_metrics, 'selected_test_mae_95_percent_group_bootstrap': interval,
            'test_mae_improvement_over_baseline': test_metrics['baseline']['mae'] - test_metrics[selected]['mae'],
            'training': training, 'versions': versions(),
            'evaluation_notes': [
                'Selection and early stopping use validation only; the reported model is not refitted on held-out data.',
                'Groups connect original/redirect registered domains and exact image hashes; visual near-duplicates and shared templates need manual review.',
                'Exact duplicate captures are retained within one group, so row counts can overstate visual diversity.',
                'The bootstrap interval resamples test groups for this fixed model, not whole training runs.',
                'Repeated experiments after reading test results weaken the meaning of a final held-out test.',
            ],
        }
        write_json(args.out / 'report.json', report)
        for split_name in ('validation', 'test'):
            indices = splits[split_name]
            prediction = candidates[selected].predict(features[indices])
            with (args.out / f'{split_name}_predictions.csv').open('w', encoding='utf-8', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['id', 'domain', 'final_url', 'group', 'your_score', 'predicted_score', 'absolute_error', 'image_path'])
                for index, value in zip(indices, prediction):
                    item = records[index]
                    writer.writerow([item.id, item.domain, item.final_url, item.group, item.score,
                                     round(float(value), 4), round(abs(item.score - float(value)), 4), str(item.path)])
        plot_report(args.out / 'report.png', scores, scores[test], test_predictions[selected], training)
        bundle = {'format_version': 1, 'encoder': encoder, 'selected': selected, 'models': candidates,
                  'feature_dimension': features.shape[1], 'versions': report['versions'],
                  'dataset_hashes': [r.sha256 for r in records]}
        temporary = args.out / 'taste_model.joblib.tmp'
        joblib.dump(bundle, temporary)
        os.replace(temporary, args.out / 'taste_model.joblib')
        print('\nHeld-out test results:')
        for name, result in test_metrics.items():
            print(f"  {name:8s} MAE={result['mae']:.3f} RMSE={result['rmse']:.3f} Spearman={result['spearman']}")
        improvement = report['test_mae_improvement_over_baseline']
        if selected == 'baseline':
            print('The baseline won validation. This run does not establish that a learned model predicts your taste.')
        elif improvement <= 0:
            print('The selected model did not beat the baseline on this test set. Inspect the mistakes before drawing conclusions.')
        else:
            print(f'Selected model reduced test MAE by {improvement:.3f} rating points relative to the baseline.')
        print(f'Saved model and evaluation in {args.out.resolve()}')
        return 0
    except KeyboardInterrupt:
        print('\nStopped. Cached embedding batches and saved splits remain reusable.', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
