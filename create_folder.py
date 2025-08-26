import os
import string

# Base dataset folder
base_dir = 'dataset'

# A-Z labels
alphabet_labels = list(string.ascii_uppercase)  # A to Z

# 0-9 digit labels
digit_labels = [str(i) for i in range(10)]  # 0 to 9

# Custom gesture labels — prefixed to push them to the end
raw_custom_labels = ['hello', 'yes', 'no', 'thumbs_up', 'ok', 'thank_you']
custom_labels = [f'custom_{label}' for label in raw_custom_labels]

# Final order: A-Z, 0-9, custom_
ordered_labels = alphabet_labels + digit_labels + custom_labels

# Create train and test folders
for split in ['train', 'test']:
    for label in ordered_labels:
        folder_path = os.path.join(base_dir, split, label)
        os.makedirs(folder_path, exist_ok=True)

print("✅ Folder structure created with custom labels at the end using 'custom_' prefix.")
