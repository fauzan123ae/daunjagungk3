import os
import shutil
import random

# Folder asal
source_dir = 'dataset'
train_dir = 'dataset/train'
validation_dir = 'dataset/validation'

split_ratio = 0.8  # 80% training, 20% validation

for category in os.listdir(source_dir):
    category_path = os.path.join(source_dir, category)
    if os.path.isdir(category_path):
        images = os.listdir(category_path)
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        os.makedirs(os.path.join(train_dir, category), exist_ok=True)
        os.makedirs(os.path.join(validation_dir, category), exist_ok=True)

        for image in train_images:
            src_path = os.path.join(category_path, image)
            dst_path = os.path.join(train_dir, category, image)
            shutil.copyfile(src_path, dst_path)

        for image in val_images:
            src_path = os.path.join(category_path, image)
            dst_path = os.path.join(validation_dir, category, image)
            shutil.copyfile(src_path, dst_path)

print('Dataset berhasil dibagi ke folder train dan validation.')
