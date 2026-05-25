from pathlib import Path
import hashlib
import json
import argparse


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(dataset_dir: Path, out_path: Path):
    files = list(dataset_dir.rglob('*'))
    images = [p for p in files if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    manifest = {
        'dataset_dir': str(dataset_dir.resolve()),
        'n_images': len(images),
        'files': [],
    }
    for p in images:
        manifest['files'].append({'path': str(p.relative_to(dataset_dir)), 'size': p.stat().st_size, 'sha256': sha256_of_file(p)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print('Wrote manifest to', out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=Path('ethiopian cofee leaf dataset'))
    parser.add_argument('--out', type=Path, default=Path('outputs/dataset_manifest.json'))
    args = parser.parse_args()
    build_manifest(args.dataset, args.out)


if __name__ == '__main__':
    main()
