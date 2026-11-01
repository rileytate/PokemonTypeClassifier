# This program determines the distribution of the pokemon type for the 
# hugging face dataset

from datasets import load_dataset, Image as dsImage
import albumentations as A
import numpy as np
from PIL import Image
import os
import pandas as pd

# create script to open files
script_dir = os.path.dirname(os.path.abspath(__file__))

# open PokemonNamesToType.csv file
pokemonCSV = pd.read_csv(os.path.join(script_dir, "PokemonNamesToType.csv"))

# open test and train files (hugging face dataset)
hugTrain = pd.read_csv(os.path.join(script_dir, "hugTrain.csv"))
hugTest = pd.read_csv(os.path.join(script_dir, "hugTest.csv"))

# combine
hugFULL = pd.concat([hugTrain, hugTest], ignore_index=True)

# lists to hold types
type1 = []
type2 = []

# index
index = 0

# iterate through hugging face dataset
for hugRow in hugFULL.itertuples():

    # iterate through csv file of pokemon to find which one it is
    for csvRow in pokemonCSV.itertuples():

        # check if hugging face matches current csv pokemon
        if(csvRow.id == hugRow.label):

            # add features to list
            type1.append(csvRow.type1)
            type2.append(csvRow.type2)

    # increment index
    index += 1

# assign list as new features
hugFULL["type1"] = type1
hugFULL["type2"] = type2


#replace null with none
hugFULL["type2"] = hugFULL["type2"].fillna("None")

# save dataframe as file
hugFULL.to_csv("HugFacePokemonTypes.csv", index=False)

print(f"Pokemon Counts Breakdown:\n{hugFULL["label"].value_counts()} \n")

print(f"Pokemon Type1 Breakdown:\n{hugFULL["type1"].value_counts()} \n")

print(f"Pokemon Type2 Breakdown:\n{hugFULL["type2"].value_counts()} \n")

#print(f"Pokemon Types Breakdown:\n{hugFULL[["type1", "type2"]].value_counts()} \n")