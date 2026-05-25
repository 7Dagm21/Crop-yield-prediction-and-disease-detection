from pathlib import Path
import sys

# Ensure project `src` is on sys.path when tests run from workspace root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.prediction.predict_disease import predict_disease


def test_predict_disease_smoke():
    project_root = Path(__file__).resolve().parents[1]
    dataset = project_root / 'ethiopian cofee leaf dataset'
    imgs = list(dataset.rglob('*.jpg')) + list(dataset.rglob('*.jpeg')) + list(dataset.rglob('*.png'))
    assert imgs, f"No images found under {dataset}"
    img = imgs[0]
    res = predict_disease(str(img), 'Coffee')
    assert isinstance(res, dict)
    assert 'disease' in res and 'confidence' in res and 'probabilities' in res
    assert isinstance(res['probabilities'], dict)
