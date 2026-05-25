from pathlib import Path
import argparse
import json
from src.training.train_disease_model import train_disease_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop', default='Coffee')
    parser.add_argument('--train_dir', type=Path, default=Path('ethiopian cofee leaf dataset')/ 'train aug')
    parser.add_argument('--test_dir', type=Path, default=Path('ethiopian cofee leaf dataset')/ 'test')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mlflow', action='store_true')
    args = parser.parse_args()

    result = train_disease_model(
        args.train_dir,
        args.test_dir,
        args.crop,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        mlflow_enabled=args.mlflow,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
