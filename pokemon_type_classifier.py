# Machine Learning
# Final Project
# Riley Tate and Harrison Niswander
# Generation One Pokemon Type Classifier

from datasets import load_dataset, Image as dsImage
import albumentations as A
import numpy as np
from PIL import Image

# Transform function explained by huggingface at 
# https://huggingface.co/docs/datasets/en/use_dataset#apply-data-augmentations
def albumentations_transforms(examples):
    # Apply Albumentations transforms
    transformed_images = []
    for image in examples["image"]:
        # Convert PIL to numpy array (OpenCV format)
        image_np = np.array(image.convert("RGB"))
        
        # Apply Albumentations transforms
        transformed_image = transform(image=image_np)["image"]
        
        # Convert back to PIL Image
        pil_image = Image.fromarray(transformed_image)
        transformed_images.append(pil_image)
    
    examples["pixel_values"] = transformed_images
    return examples

# Load the entire dataset
dataset = load_dataset("Dusduo/1stGen-Pokemon-Images")

# Make image in RBG mode
dataset = dataset.cast_column("image", dsImage(mode="RGB"))

# Transform the Images
transform = A.Compose([
    A.RandomCrop(height=256, width=256, pad_if_needed=True, p=1),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.5)
])

dataset = dataset.with_transform(albumentations_transforms)

# Split so some pokemon are not present in training data