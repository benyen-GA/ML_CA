import os
import random
from PIL import Image

input_root = 'train_2'
output_mix_path = os.path.join(input_root,'mix')
os.makedirs(output_mix_path,exist_ok=True)

combinations_per_size  = 100
target_size = (224,224)
max_images_per_mix = 4

class_image_paths = {}
for class_name in os.listdir(input_root):
    class_path = os.path.join(input_root, class_name)
    if not os.path.isdir(class_path) or class_name == 'mix':
        continue
    class_image_paths[class_name] = [
        os.path.join(class_path, f)
        for f in os.listdir(class_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]

mix_count = 0
for num_imgs in range(2, max_images_per_mix + 1):
    for _ in range(combinations_per_size):
        selected_classes = []
        selected_paths = []

        for _ in range(num_imgs):
            attempts = 0
            while attempts < 10:
                cls = random.choice(list(class_image_paths.keys()))
                if num_imgs == 2 and cls in selected_classes:
                    attempts += 1
                    continue
                img_path = random.choice(class_image_paths[cls])
                selected_classes.append(cls)
                selected_paths.append(img_path)
                break

        if len(selected_paths) != num_imgs:
            continue

        images = []
        for path in selected_paths:
            try:
                img = Image.open(path).convert('RGB')
                images.append(img)
            except Exception as e:
                print(f"skip damage image: {path}")
                continue

        if len(images) != num_imgs:
            continue

        if random.random() < 0.5:
            total_width = sum(img.width for img in images)
            max_height = max(img.height for img in images)
            result = Image.new('RGB', (total_width, max_height), (255, 255, 255))
            x_offset = 0
            for img in images:
                result.paste(img, (x_offset, 0))
                x_offset += img.width
        else:
            total_height = sum(img.height for img in images)
            max_width = max(img.width for img in images)
            result = Image.new('RGB', (max_width, total_height), (255, 255, 255))
            y_offset = 0
            for img in images:
                result.paste(img, (0, y_offset))
                y_offset += img.height
        
        # result = result.resize(target_size, Image.Resampling.LANCZOS)

        save_name = f"mix_{num_imgs}_{mix_count}.jpg"
        result.save(os.path.join(output_mix_path, save_name))
        mix_count += 1

print(f" {mix_count} mix images are generated, saved in {output_mix_path}")