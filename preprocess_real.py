"""
2. REAL Preprocessing Pipeline
Bilateral Filter → CLAHE → Resize → Normalize
"""
import cv2
import numpy as np
import os
from PIL import Image
from tqdm import tqdm # type: ignore

classes = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'no_tumor']
input_dir = 'multi_tumor_dataset/Training'
output_dir = 'processed_dataset/Training'

# Create output folders
for cls in classes:
    os.makedirs(f"{output_dir}/{cls}", exist_ok=True)

print("🛠️  REAL PREPROCESSING PIPELINE")
print("="*50)

for cls in classes:
    input_folder = f"{input_dir}/{cls}"
    output_folder = f"{output_dir}/{cls}"
    
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in tqdm(images, desc=f"[{cls}]", leave=False):
        # 1. Load
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        if img is None: continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. NOISE REDUCTION (Bilateral Filter)
        denoised = cv2.bilateralFilter(img_rgb, 15, 80, 80)
        
        # 3. CONTRAST (CLAHE per channel)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = np.zeros_like(denoised)
        for c in range(3):
            gray = cv2.cvtColor(denoised, cv2.COLOR_RGB2GRAY)
            enhanced[:, :, c] = clahe.apply(gray)
        
        # 4. RESIZE 224x224 (ResNet standard)
        resized = cv2.resize(enhanced, (224, 224))
        
        # 5. SAVE PROCESSED
        output_path = os.path.join(output_folder, img_name)
        Image.fromarray(resized.astype(np.uint8)).save(output_path)
    
    print(f"✅ {cls}: {len(images)} images processed")

print("\n🎉 PROCESSED DATASET READY!")
print("📁 Location: processed_dataset/Training/")
