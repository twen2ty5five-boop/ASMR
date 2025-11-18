import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# =========================================================
# 1. 데이터 로드 & 전처리 (MNIST)
# =========================================================
transform = transforms.Compose([
    transforms.ToTensor(),                     # [0,255] → [0,1]
    transforms.Normalize((0.1307,), (0.3081,)) # MNIST 평균/표준편차로 정규화
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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# =========================================================
# 4. 학습 루프 + 그래프용 기록 리스트
# =========================================================
num_epochs = 3
train_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # ----- Train -----
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
    train_losses.append(avg_loss)

    # ----- Eval -----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    epoch_acc = correct / total
    test_accuracies.append(epoch_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_loss:.4f} - Test Accuracy: {epoch_acc:.4f}")

# =========================================================
# 5. 최종 평가 결과 출력
# =========================================================
print("Final Test Accuracy:", test_accuracies[-1])

# =========================================================
# 6. 모델 저장 & 로드
# =========================================================
save_path = "simple_cnn_mnist.pth"
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")

loaded_model = SimpleCNN().to(device)
loaded_model.load_state_dict(torch.load(save_path, map_location=device))
loaded_model.eval()
print("Loaded model is ready for inference.")

# =========================================================
# 7. 그래프 그리기 (Loss & Accuracy)
# =========================================================
epochs = range(1, num_epochs + 1)

plt.figure()
plt.plot(epochs, train_losses, marker="o")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

plt.figure()
plt.plot(epochs, test_accuracies, marker="o")
plt.title("Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
