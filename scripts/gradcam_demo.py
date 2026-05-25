from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))

import torch
from src.prediction.predict_disease import DiseaseClassifier
from src.preprocessing.image_preprocessing import build_image_transforms
from src.explainability.grad_cam import GradCAM, save_gradcam_overlay


def main():
    root = Path.cwd()
    model_path = root / 'models' / 'disease' / 'coffee_disease_model.pth'
    test_img = root / 'ethiopian cofee leaf dataset' / 'test' / 'Cerscospora'
    imgs = list(test_img.glob('*.jpg'))
    if not imgs:
        print('NO_IMAGE')
        return
    img = imgs[0]

    bundle = torch.load(model_path, map_location='cpu')
    class_names = bundle['class_names']
    image_size = bundle.get('image_size', 224)

    model = DiseaseClassifier(num_classes=len(class_names))
    model.load_state_dict(bundle['model_state_dict'])
    model.eval()

    transform = build_image_transforms(image_size=image_size, training=False)
    from PIL import Image
    img_pil = Image.open(img).convert('RGB')
    x = transform(img_pil).unsqueeze(0)

    cam = GradCAM(model)
    heatmap = cam.generate_cam(x)

    out = root / 'outputs' / 'gradcam' / (img.stem + '_gradcam.png')
    save_gradcam_overlay(img, heatmap, out)
    print('Saved', out)


if __name__ == '__main__':
    main()
