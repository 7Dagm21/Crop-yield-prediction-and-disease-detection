from pathlib import Path
import shutil
import json

from src.config import normalize_crop_name, get_disease_model_path


def copy_and_normalize(src_dir: Path, dst_root: Path, crop: str) -> Path:
    norm = normalize_crop_name(crop)
    dst_crop_dir = dst_root / norm
    dst_crop_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if not item.is_dir():
            continue
        name = item.name
        if name.lower() == "healthy":
            dst_name = "healthy"
        else:
            safe = name.lower().replace(" ", "_")
            dst_name = f"{norm}_{safe}"
        dst_path = dst_crop_dir / dst_name
        if dst_path.exists():
            shutil.rmtree(dst_path)
        shutil.copytree(item, dst_path)
    return dst_crop_dir


def main():
    root = Path(__file__).resolve().parents[1]
    src_root = root / "ethiopian cofee leaf dataset"
    train_src = src_root / "train aug"
    test_src = src_root / "test"

    dst_train_root = root / "data" / "disease_images" / "train"
    dst_test_root = root / "data" / "disease_images" / "test"

    crop = "Coffee"

    if not train_src.exists():
        raise FileNotFoundError(f"Expected training folder at {train_src}")

    print("Copying and normalizing training folders...")
    train_dst = copy_and_normalize(train_src, dst_train_root, crop)

    if test_src.exists():
        print("Copying and normalizing test folders...")
        test_dst = copy_and_normalize(test_src, dst_test_root, crop)
    else:
        test_dst = dst_test_root / normalize_crop_name(crop)

    print("Imported dataset to:")
    print("  train:", train_dst)
    print("  test:", test_dst)

    # run a short training run using internal API
    try:
        from src.training.train_disease_model import train_disease_model

        print("Starting short training (3 epochs)...")
        res = train_disease_model(train_dst, test_dst, crop, epochs=3, batch_size=16, patience=2)
        print(json.dumps(res, indent=2))
    except Exception as e:
        print("Training failed:", e)


if __name__ == "__main__":
    main()
