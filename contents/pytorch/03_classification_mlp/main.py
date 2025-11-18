import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 다음 코드는 다층 퍼셉트론(MLP)으로 기본 분류(Classification) 모델을 만들 때 사용합니다.
# MNIST 숫자 분류의 가장 기본 구조를 PyTorch로 구현한 코드입니다.

# MNIST 데이터 로드 + 전처리
transform = transforms.Compose([
    transforms.ToTensor(),               # (H,W,C) → (C,H,W) + 0~1 스케일
    transforms.Lambda(lambda x: x.view(-1))  # 28×28 → 784 벡터로 reshape
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# MLP 모델 정의
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
    nn.LogSoftmax(dim=1)   # PyTorch는 CrossEntropyLoss 내부에서 LogSoftmax+NLLLoss 처리 가능
)

# 손실함수 & 옵티마이저
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------- 학습(Training) --------
for epoch in range(3):
    for images, labels in train_loader:
        # Forward
        pred = model(images)
        loss = criterion(pred, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

# -------- 평가(Evaluation) --------
correct = 0
total = 0
model.eval()
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print("Test Accuracy:", correct / total)
