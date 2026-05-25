from pathlib import Path
import json
import datetime


def generate_model_card(bundle_path: Path, dataset_manifest: Path | None = None, out_path: Path | None = None):
    bundle = json.loads(bundle_path.read_bytes()) if bundle_path.suffix == '.json' else None
    # try torch load lazily to avoid heavy imports; if it's a torch bundle, read minimal keys via torch
    try:
        import torch
        b = torch.load(bundle_path, map_location='cpu')
    except Exception:
        b = bundle or {}

    class_names = b.get('class_names', [])
    metrics = b.get('metrics', {})
    image_size = b.get('image_size', 224)

    model_name = bundle_path.stem
    out_path = out_path or (bundle_path.parent / (model_name + '_model_card.md'))

    lines = [f'# Model Card — {model_name}', '', f'Generated: {datetime.datetime.utcnow().isoformat()} UTC', '']
    lines += ['## Summary', f'- Model path: {bundle_path}', f'- Number of classes: {len(class_names)}', f'- Image size: {image_size}', '']
    if class_names:
        lines += ['## Classes', '']
        for c in class_names:
            lines.append(f'- {c}')
        lines.append('')

    lines += ['## Metrics', '']
    if metrics:
        for k, v in metrics.items():
            lines.append(f'- {k}: {v}')
    else:
        lines.append('- No metrics embedded in bundle')

    if dataset_manifest and dataset_manifest.exists():
        dm = json.loads(dataset_manifest.read_text())
        lines += ['', '## Dataset', f'- Path: {dm.get("dataset_dir")}', f'- n_images: {dm.get("n_images")}']

    out_path.write_text('\n'.join(lines))
    print('Wrote model card to', out_path)


if __name__ == '__main__':
    import sys
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('models/disease/coffee_disease_model.pth')
    dm = Path('outputs/dataset_manifest.json')
    generate_model_card(p, dm)
