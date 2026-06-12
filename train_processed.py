"""
3. Train ResNet18 - MAC OMP FIXED
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # ✅ FIXES OMP ERROR

import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# Dataset Class
classes = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'no_tumor']

class ProcessedDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.images = []
        self.labels = []
        
        for idx, cls in enumerate(classes):
            cls_path = os.path.join(root_dir, cls)
            if os.path.exists(cls_path):
                for img_name in os.listdir(cls_path):
                    self.images.append(os.path.join(cls_path, img_name))
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        return self.transform(image), label

def main():
    print("📊 Loading PROCESSED Dataset...")
    dataset = ProcessedDataset('processed_dataset/Training')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    device = torch.device('cpu')
    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, 4)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("🤖 Training Multi-Class Model...")
    print("Epoch | Loss | Accuracy")
    print("-" * 30)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"{epoch+1:2d}    | {running_loss/len(dataloader):6.3f} | {accuracy:5.1f}%")

    torch.save(model.state_dict(), 'multi_tumor_processed.pth')
    print("\n✅ Model Saved: multi_tumor_processed.pth")
    print("🎯 Ready for Flask deployment!")

if __name__ == '__main__':
    main()
