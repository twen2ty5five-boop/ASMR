import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 다음 코드는 합성곱 신경망(CNN)으로 MNIST 숫자 이미지를 분류할 때 사용합니다.
# 텐서플로우 버전의 Conv2D → MaxPooling2D → Conv2D → Flatten → Dense(10) 구조를
# PyTorch로 그대로 구현한 기본 예제입니다.

# ===========================
# 1. 데이터 로드 & 전처리
# ===========================
transform = transforms.Compose([
    transforms.ToTensor(),          # [0,255] → [0,1], (H,W) → (1,H,W)
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ===========================
# 2. CNN 모델 정의
# ===========================
class SimpleCNN(nn.Module):
    # TF:
    # Conv2D(32, 3) → MaxPooling2D() → Conv2D(64, 3) → Flatten → Dense(10)
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)     # (1,28,28) → (32,26,26)
        self.pool  = nn.MaxPool2d(2, 2)                  # (32,26,26) → (32,13,13)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)    # (32,13,13) → (64,11,11)
        self.fc    = nn.Linear(64 * 11 * 11, 10)         # Flatten 후 전결합층

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x  # 최종 출력(logits), CrossEntropyLoss와 함께 사용

model = SimpleCNN()

# ===========================
# 3. 손실함수 & 옵티마이저
# ===========================
criterion = nn.CrossEntropyLoss()           # TF의 sparse_categorical_crossentropy와 동일
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===========================
# 4. 학습 루프 (epochs=3)
# ===========================
for epoch in range(3):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

# ===========================
# 5. 평가 (Test Accuracy)
# ===========================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)                 # (B,10)
        _, predicted = torch.max(outputs, 1)    # 가장 큰 값의 인덱스 = 예측 클래스
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print("Test Accuracy:", correct / total)
