import os

base_dir = "id_card_project"
sub_dirs = [
    "images",          # Store input ID card images
    "masks",           # Store corresponding binary masks
    "dataset/train/images",
    "dataset/train/masks",
    "dataset/val/images",
    "dataset/val/masks",
    "models",          # For saving trained models
    "outputs",         # For prediction outputs
    "scripts"          # For all training/testing scripts
]

for sub_dir in sub_dirs:
    path = os.path.join(base_dir, sub_dir)
    os.makedirs(path, exist_ok=True)

print(f" Project folders created in: {base_dir}")
