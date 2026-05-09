# Social Informatics — Brain Tumor Image Classifier
how to load the dataset from drive to colab and run the project

cell 1

from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive mounted!")

---------------------------------
cell 2

import os
dataset_path = "/content/drive/MyDrive/Colab Notebooks/brain_tumor_dataset"
for folder in os.listdir(dataset_path):
    count = len(os.listdir(f"{dataset_path}/{folder}"))
    print(f"  📁 {folder}: {count} images")

    ------------------------------
cell 3

    import os
dataset_path = "/content/drive/MyDrive/Colab Notebooks/brain_tumor_dataset"

if os.path.exists(dataset_path):
    print("✅ Dataset folder found!")
    for folder in os.listdir(dataset_path):
        full_path = f"{dataset_path}/{folder}"
        if os.path.isdir(full_path):
            count = len(os.listdir(full_path))

            ------------------------
cell 4

            # Read the classifier code
with open('brain_tumor_classifier_colab.py', 'r') as f:
    code = f.read()


code = code.replace(
    '"brain_tumor_dataset"',
    '"/content/drive/MyDrive/Colab Notebooks/brain_tumor_dataset"'
)


if '/content/drive/MyDrive/Colab Notebooks/brain_tumor_dataset' in code:
    print("✅ Path replaced successfully!")
else:
    print("❌ Path replacement failed!")


exec(code)
            print(f"  📁 {folder}: {count} images")
else:
    print("❌ Path not found — check your Drive path")
