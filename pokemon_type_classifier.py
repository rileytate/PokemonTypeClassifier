# Machine Learning
# Final Project
# Riley Tate and Harrison Niswander
# Generation One Pokemon Type Classifier

from datasets import load_dataset, Image as dsImage
import albumentations as A
import numpy as np
from PIL import Image
import os
import pandas as pd

# create script to open files
script_dir = os.path.dirname(os.path.abspath(__file__))

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

# function to add new features (pokemon type) to dataset
def addTypeFeature(dataset):
    # open PokemonNamesToType.csv file
    pokemonCSV = pd.read_csv(os.path.join(script_dir, "PokemonNamesToType.csv"))

    # Mapping function that queries the pandas DataFrame directly
    def map_types(example):
        pokemon_name = example['label']
        
        # Look up the row in the CSV where the 'name' column matches the pokemon's name
        matching_rows = pokemonCSV.loc[pokemonCSV['name'] == pokemon_name]
        
        if not matching_rows.empty:
            # .iloc[0] grabs the first match found
            example['type1'] = matching_rows['type1'].iloc[0]
            example['type2'] = matching_rows['type2'].iloc[0]

        else:
            # Fallback values if the name isn't found in your CSV
            example['type1'] = None
            example['type2'] = None
            
        return example

    # Use Hugging Face's map method to apply it to all rows safely
    updated_dataset = dataset.map(map_types)
    return updated_dataset

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

# add new features to dataset (pokemon's type)
dataset = addTypeFeature(dataset)
print(dataset)

# apply image albumentations tranformation
dataset = dataset.with_transform(albumentations_transforms)



# Split so some pokemon are not present in training data