from collections import defaultdict
import os

valid_extensions = ('.jpg', '.jpeg', '.png')  # Add more if needed

train_path = "../shawn/train"
class_counts = defaultdict(int)

for class_name in os.listdir(train_path):
    class_dir = os.path.join(train_path, class_name)
    if os.path.isdir(class_dir):
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(valid_extensions):
                class_counts[class_name] += 1

print("Updated class distribution:")
for cls, count in class_counts.items():
    print(f"{cls}: {count} image files")
print("Absolute path:", os.path.abspath(train_path))
