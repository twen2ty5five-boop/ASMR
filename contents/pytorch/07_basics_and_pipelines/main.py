import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 다음 코드는 PyTorch로 딥러닝 기본 파이프라인 전체를 한 번에 구성하는 예제입니다.
# (데이터 로드 → DataLoader → 모델 정의 → 손실함수/옵티마이저 → 학습 → 평가 → 모델 저장)의
# 전체 흐름(basics_and_pipelines)을 이해하는 데 사용합니다.

# =========================================================
# 1. 데이터 로드 & 전처리 (MNIST)
# =========================================================
transform = transforms.Compose([
    transforms.ToTensor(),                     # [0,255] → [0,1]
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST 평균/표준편차로 정규화 (선택)
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=2,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=2,
)

# =========================================================
# 2. 모델 정의 (간단한 CNN)
#    입력: (1, 28, 28) → 출력: 10 클래스
# =========================================================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)   # (1,28,28) → (32,26,26)
        self.pool  = nn.MaxPool2d(2, 2)               # (32,26,26) → (32,13,13)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # (32,13,13) → (64,11,11)
        self.fc1   = nn.Linear(64 * 11 * 11, 128)
        self.fc2   = nn.Linear(128, 10)               # 10개 숫자 분류

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)   # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)             # logits
        return x

model = SimpleCNN()

# =========================================================
# 3. 손실함수 & 옵티마이저
# =========================================================
criterion = nn.CrossEntropyLoss()                  # 다중 클래스 분류
optimizer = optim.Adam(model.parameters(), lr=0.001)

# GPU 사용 가능하면 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================================================
# 4. 학습 루프 (Training)
# =========================================================
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f}")

# =========================================================
# 5. 평가 루프 (Evaluation)
# =========================================================
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)           # (batch, 10)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print("Test Accuracy:", test_acc)

# =========================================================
# 6. 모델 저장 & 로드 (파이프라인 마무리)
# =========================================================
save_path = "simple_cnn_mnist.pth"
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")

# 새 인스턴스를 만들어서 로드
loaded_model = SimpleCNN().to(device)
loaded_model.load_state_dict(torch.load(save_path, map_location=device))
loaded_model.eval()
print("Loaded model is ready for inference.")
