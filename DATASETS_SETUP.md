# Dataset Setup Guide

This project is scoped to Ethiopian grains and coffee only:

- Teff
- Wheat
- Maize
- Sorghum
- Barley
- Millet
- Chickpea
- Coffee

## 1. Yield Data

### Recommended sources

- FAOSTAT: https://www.fao.org/faostat/
- NASA POWER climate data: https://power.larc.nasa.gov/
- World Bank climate indicators: https://data.worldbank.org/
- Ethiopian Ministry of Agriculture data, if available

### Expected yield features

- crop_type
- year
- region
- rainfall_mm
- temperature_c
- humidity
- fertilizer_kg_per_ha
- area_harvested_ha
- yield_kg_per_ha

### Current project data

The repository includes a starter Ethiopia yield table at:

- data/crop_yield/faostat_ethiopia_yield.csv

## 2. Disease Data

### Current disease scope

- Coffee
- Wheat
- Maize
- Sorghum

### Future disease work

- Teff

Barley, Millet, and Chickpea remain yield crops in this project, but a reliable public disease image set was not identified in the workspace for them yet.

### Folder structure

```text
data/disease_images/
├── train/
│   ├── coffee_healthy/
│   ├── coffee_leaf_rust/
│   ├── coffee_berry_disease/
│   ├── wheat_healthy/
│   ├── wheat_leaf_rust/
│   ├── wheat_powdery_mildew/
│   ├── maize_healthy/
│   ├── maize_common_rust/
│   ├── maize_leaf_blight/
│   ├── sorghum_healthy/
│   ├── sorghum_anthracnose/
│   ├── sorghum_rust/
│   └── sorghum_leaf_spot/
└── test/
    └── same class structure as train/
```

### Recommended disease classes

- coffee_healthy
- coffee_leaf_rust
- coffee_berry_disease
- wheat_healthy
- wheat_leaf_rust
- wheat_powdery_mildew
- maize_healthy
- maize_common_rust
- maize_leaf_blight
- sorghum_healthy
- sorghum_anthracnose
- sorghum_rust
- sorghum_leaf_spot

### Primary public sources

- Coffee: https://www.kaggle.com/datasets/anujms/coffee-leaf-disease
- Wheat: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- Maize: https://www.kaggle.com/datasets/emmarex/plantdisease
- Sorghum: https://universe.roboflow.com/search?q=sorghum+disease
- Teff: no large public disease dataset identified yet

### Notes

- Use crop-specific image datasets whenever possible.
- The training code uses ResNet18 transfer learning.
- The model expects folders arranged for `ImageFolder`.

## 3. Download workflow

1. Download or export the yield table.
2. Place the CSV in `data/crop_yield/`.
3. Collect disease images and arrange them in `data/disease_images/train` and `data/disease_images/test`.
4. Run the training commands:

```bash
python -m src.training.train_yield_model
python -m src.training.train_disease_model
```

## 4. If you use Kaggle

- Create a Kaggle API token from: https://www.kaggle.com/settings/account
- Save it to `C:\Users\<username>\.kaggle\kaggle.json`
- Run:

```bash
python setup_datasets.py
```

## 5. Validation

After training, confirm the following outputs exist:

- `models/yield_regression_model.pth`
- `models/preprocessor.pkl`
- `models/disease/coffee_disease_model.pth`
- `models/disease/wheat_disease_model.pth`
- `models/disease/maize_disease_model.pth`
- `models/disease/sorghum_disease_model.pth`
- `outputs/reports/yield_metrics.json`
- `outputs/reports/disease_metrics_coffee.json`
- `outputs/reports/disease_metrics_wheat.json`
- `outputs/reports/disease_metrics_maize.json`
- `outputs/reports/disease_metrics_sorghum.json`
- `outputs/graphs/yield_training_curves.png`
- `outputs/graphs/disease_confusion_matrix_coffee.png`
- `outputs/graphs/disease_confusion_matrix_wheat.png`
- `outputs/graphs/disease_confusion_matrix_maize.png`
- `outputs/graphs/disease_confusion_matrix_sorghum.png`

## 6. Practical recommendation

For the cleanest final system, keep the yield model on Ethiopian crop + climate data, and train the disease model only on crop-specific datasets that match the supported crops above.
