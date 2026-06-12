"""
1. Download Kaggle Multi-Class Brain Tumor Dataset
"""
import os
import subprocess
import zipfile

print("📥 DOWNLOADING MULTI-CLASS DATASET")
print("="*50)

# Download dataset
subprocess.run(["pip", "install", "kaggle"], shell=True)
subprocess.run([
    "kaggle", "datasets", "download", 
    "-d", "masoudnickparvar/brain-tumor-mri-dataset"
], check=True)

# Extract
with zipfile.ZipFile("brain-tumor-mri-dataset.zip", 'r') as zip_ref:
    zip_ref.extractall("multi_tumor_dataset/")

print("✅ Dataset downloaded: multi_tumor_dataset/")
print("✅ Classes: glioma_tumor, meningioma_tumor, pituitary_tumor, no_tumor")
