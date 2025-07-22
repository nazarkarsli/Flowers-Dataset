import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as utils
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


batch_size=32
image_size=28*28
transform=transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
])
data_dir = "\Kendi_Kodlarim\data\flowers"
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size) #%80 i eğitim %20si test
test_size = dataset_size - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train örnek sayısı: {len(train_dataset)}")
print(f"Test örnek sayısı: {len(test_dataset)}")
print(f"Sınıflar: {full_dataset.classes}")
#3 128 → 64 →32 →16
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv2d(3,16,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        #fully-connected katman:
        #hesaplama:
        #girdi : 128x128
        #conv1 + pool => 64x64 (channels =32)
        #conv2 + pool => 32x32 (channels=64)
        #conv3 + pool => 16x16 chan 128
        #toplam flatten boyutu 128 * 16 * 16
        self.fc1 = nn.Linear(64 * 16 * 16 , 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)  #overfittingi önlemek için dropout kullanıyoruz
    def forward(self, x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.conv3(x)))
        x=x.view(x.size(0),-1) #flatten

        x=self.dropout(F.relu(self.fc1(x)))
        x=self.fc2(x)
        return x

device = torch.device("cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct, total = 0.0, 0.0
    for inputs,labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct+= (predicted == labels).sum().item()
        total+= labels.size(0)
    acc= 100 * correct / total
    print(f"Epoch {epoch} Loss: {running_loss/len(train_loader)}")

model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs,labels in test_loader:
        inputs,labels=inputs.to(device),labels.to(device)
        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)
        total += labels.size(0)
        correct+= (predictions == labels).sum().item()
    print(f"Accuracy: {(100*correct/total)}")

    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    class_names = full_dataset.classes
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Model Tahminleri", fontsize=18)

    for idx in range(8):
        ax = axes[idx // 4, idx % 4] #2x4 grid içinden doğru konumu seçer
        img = images[idx].cpu().permute(1, 2, 0)
       #normalizeden geri döndürüyor -1 ile 1 arasına çekmiştik şimdi renkler doğal gözüksün diye 0 ile 1 arasına
        img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # unnormalize
        img = img.clamp(0, 1)
        ax.imshow(img)
        pred_label = class_names[preds[idx]]
        true_label = class_names[labels[idx]]
        color = "green" if pred_label == true_label else "red"
        ax.set_title(f"Gerçek: {true_label}\nTahmin: {pred_label}", color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
