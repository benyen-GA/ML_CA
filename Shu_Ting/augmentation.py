import os
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F
import random

augmentation = transforms.Compose([
    transforms.RandomApply([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
    ], p=0.5),

    transforms.RandomApply([
        transforms.RandomRotation(degrees=30),
    ], p=0.8),

    transforms.RandomApply([
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),  
            scale=(0.95, 1.05),    
            shear=5               
        ),
    ], p=0.7),

    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.02
        ),
    ], p=0.8),

    transforms.RandomApply([
        transforms.RandomGrayscale(p=1.0),
    ], p=0.2),

    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3),
    ], p=0.3),
])

input_root = 'train_1'
output_root = 'augmented_train_1'
aug_per_image = 5

for class_name in os.listdir(input_root):
    input_class_path = os.path.join(input_root, class_name)
    output_class_path = os.path.join(output_root, class_name)

    if not os.path.isdir(input_class_path):
        continue

    os.makedirs(output_class_path, exist_ok=True)

    for img_name in os.listdir(input_class_path):
        img_path = os.path.join(input_class_path, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"skip damage image: {img_path}")
            continue

        for i in range(aug_per_image):
            aug_img = augmentation(image)
            save_path = os.path.join(
                output_class_path,
                f"{os.path.splitext(img_name)[0]}_aug{i}.jpg"
            )
            aug_img.save(save_path)