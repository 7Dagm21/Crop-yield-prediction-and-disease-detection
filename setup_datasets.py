"""
Setup script for Ethiopian crop intelligence datasets.

This project is scoped to Ethiopian grains and coffee:
Teff, Wheat, Maize, Coffee, Sorghum, Barley, Millet, Chickpea.

Run this after placing kaggle.json in ~/.kaggle/.

Usage:
    python setup_datasets.py --yield-only
    python setup_datasets.py --disease-only
    python setup_datasets.py
"""

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

# Dataset configurations
KAGGLE_DATASETS = {
    "yield_comprehensive": {
        "id": "abdallahalidev/agri-crop-yield-prediction",
        "description": "Comprehensive crop yield data",
        "crops": ["Wheat", "Maize", "Barley", "Sorghum", "Chickpea"],
    },
    "fao_stat": {
        "url": "https://www.fao.org/faostat",
        "description": "FAO official data - requires manual download and registration",
        "crops": ["Teff", "Wheat", "Maize", "Coffee", "Sorghum", "Barley", "Millet", "Chickpea"],
    },
    "coffee_disease": {
        "id": "anujms/coffee-leaf-disease",
        "description": "Coffee leaf disease images",
        "crops": ["Coffee"],
    },
    "corn_disease": {
        "id": "emmarex/plantdisease",
        "description": "PlantVillage maize disease images",
        "crops": ["Maize"],
    },
    "wheat_disease": {
        "id": "vipoooool/new-plant-diseases-dataset",
        "description": "Wheat disease images",
        "crops": ["Wheat"],
    },
    "sorghum_disease": {
        "url": "https://universe.roboflow.com/search?q=sorghum+disease",
        "description": "Sorghum disease images on Roboflow",
        "crops": ["Sorghum"],
    },
}

CROPS = ["Teff", "Wheat", "Maize", "Coffee", "Sorghum", "Barley", "Millet", "Chickpea"]

def check_kaggle_auth() -> bool:
    """Check if Kaggle API is configured."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        print("❌ Kaggle API not configured.")
        print(f"   Place your kaggle.json at: {kaggle_json}")
        print("   Download from: https://www.kaggle.com/settings/account")
        return False
    return True


def download_kaggle_dataset(dataset_id: str, output_dir: Path) -> bool:
    """Download a dataset from Kaggle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"   Downloading {dataset_id}...")
        if shutil.which("kaggle") is None:
            raise FileNotFoundError(
                "The Kaggle CLI was not found on PATH. Install kaggle and place kaggle.json in your ~/.kaggle directory."
            )
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(output_dir)],
            check=True,
            capture_output=True,
        )
        # Unzip if needed
        for zip_file in output_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_file, "r") as archive:
                archive.extractall(output_dir)
            zip_file.unlink()
        print(f"   ✅ Downloaded to {output_dir}")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def setup_yield_data() -> bool:
    """Download and prepare yield datasets."""
    print("\n📊 Setting up Yield Data...")
    yield_dir = Path("data/crop_yield/raw")
    
    if not check_kaggle_auth():
        print("   ⚠️  Skipping Kaggle datasets. Download manually from:")
        for name, config in KAGGLE_DATASETS.items():
            if "yield" in name or "fao" in name:
                print(f"      - {config['description']}")
                if "url" in config:
                    print(f"        {config['url']}")
                elif "id" in config:
                    print(f"        https://kaggle.com/datasets/{config['id']}")
        return False
    
    # Download main yield dataset
    for name, config in KAGGLE_DATASETS.items():
        if "yield" in name and "id" in config:
            download_kaggle_dataset(config["id"], yield_dir)
    
    print("   ✅ Yield data setup complete")
    return True


def setup_disease_data() -> bool:
    """Download and prepare disease image datasets."""
    print("\n🌾 Setting up Disease Image Data...")
    
    if not check_kaggle_auth():
        print("   ⚠️  Skipping. Download crop-specific datasets manually:")
        for name, config in KAGGLE_DATASETS.items():
            if "disease" in name:
                print(f"      - {config['description']}")
                if "id" in config:
                    print(f"        https://kaggle.com/datasets/{config['id']}")
                elif "url" in config:
                    print(f"        {config['url']}")
        return False
    
    base_dir = Path("data/disease_images")
    
    # Download crop-specific disease datasets
    disease_datasets = {
        "Coffee": KAGGLE_DATASETS["coffee_disease"],
        "Maize": KAGGLE_DATASETS["corn_disease"],
        "Wheat": KAGGLE_DATASETS["wheat_disease"],
        "Sorghum": KAGGLE_DATASETS["sorghum_disease"],
    }
    
    for crop, config in disease_datasets.items():
        print(f"\n   {crop}:")
        if "id" in config:
            crop_dir = base_dir / crop.lower()
            download_kaggle_dataset(config["id"], crop_dir)
    
    print("\n   ✅ Disease data setup complete")
    return True


def generate_combine_script() -> None:
    """Generate a script to combine downloaded datasets into training format."""
    script_content = '''"""
Combine and organize downloaded datasets into training/testing folders.
"""
from pathlib import Path
import shutil
import random

def organize_disease_images():
    """Organize images into data/disease_images/train and test folders."""
    base = Path("data/disease_images")
    train_dir = base / "train"
    test_dir = base / "test"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    for crop_dir in base.glob("*/"):
        if crop_dir.name in ("train", "test"):
            continue
        
        crop_name = crop_dir.name
        print(f"Processing {crop_name}...")
        
        # Find all image subdirectories or images
        images = list(crop_dir.rglob("*.jpg")) + list(crop_dir.rglob("*.png"))
        if not images:
            continue
        
        random.shuffle(images)
        split = int(0.8 * len(images))
        
        for img in images[:split]:
            class_name = img.parent.name
            target = train_dir / crop_name / class_name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, target / img.name)
        
        for img in images[split:]:
            class_name = img.parent.name
            target = test_dir / crop_name / class_name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, target / img.name)
        
        print(f"  ✅ {len(images)} images organized for {crop_name}")

if __name__ == "__main__":
    organize_disease_images()
'''
    
    script_path = Path("scripts/organize_datasets.py")
    script_path.parent.mkdir(exist_ok=True)
    script_path.write_text(script_content)
    print(f"   ✅ Generated {script_path}")


def print_next_steps() -> None:
    """Print instructions for the user."""
    print("\n" + "="*60)
    print("📋 NEXT STEPS:")
    print("="*60)
    print("""
1. **Get Kaggle API credentials:**
   - Go to https://www.kaggle.com/settings/account
   - Click "Create New API Token"
   - This downloads kaggle.json
   - Place it in ~/.kaggle/kaggle.json

2. **Run this script again:**
   python setup_datasets.py

3. **Organize downloaded images (after downloads complete):**
   python scripts/organize_datasets.py

4. **Retrain models with new data:**
   python -m src.training.train_yield_model
   python -m src.training.train_disease_model

    Disease training is currently grounded in Coffee, Wheat, Maize, and Sorghum. Teff remains future work until a real image set is available.

5. **Start the dashboard:**
   streamlit run main.py
    """)


def main():
    parser = argparse.ArgumentParser(description="Setup datasets for Ethiopian Crop Intelligence System")
    parser.add_argument("--yield-only", action="store_true", help="Download yield data only")
    parser.add_argument("--disease-only", action="store_true", help="Download disease images only")
    args = parser.parse_args()
    
    print("🌾 Ethiopian Crop Intelligence System - Dataset Setup")
    print("="*60)
    
    success = True
    if not args.disease_only:
        success = setup_yield_data() and success
    
    if not args.yield_only:
        success = setup_disease_data() and success
    
    if success:
        generate_combine_script()
        print_next_steps()
    else:
        print("\n⚠️  Some datasets could not be downloaded automatically.")
        print("    Please configure Kaggle API or download manually from:")
        for name, config in KAGGLE_DATASETS.items():
            print(f"    - {config['description']}")


if __name__ == "__main__":
    main()
