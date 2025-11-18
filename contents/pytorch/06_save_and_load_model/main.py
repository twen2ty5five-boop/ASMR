06_save_and_load_model

import torch
import torch.nn as nn
import torch.optim as optim

# 다음 코드는 학습한 PyTorch 모델을 저장하고, 다시 로드하여 사용하는 기본 구조를 보여줍니다.
# 간단한 MLP 모델을 정의하고, state_dict 형식으로 저장/로드하는 예제입니다.

# ===========================
# 1. 간단한 모델 정의
# ===========================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# (선택) 더미 데이터로 간단히 한 번 학습해보기
x_dummy = torch.randn(16, 10)     # 배치 16, 입력 차원 10
y_dummy = torch.randn(16, 1)      # 타깃

model.train()
pred = model(x_dummy)
loss = criterion(pred, y_dummy)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Before saving - sample loss:", loss.item())

# ===========================
# 2. 모델 저장
# ===========================
# 일반적으로 model.state_dict()만 저장하는 것이 권장되지만,
# 여기서는 optimizer까지 함께 저장하는 예제를 보여줍니다.
checkpoint_path = "saved_model.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}, checkpoint_path)

print(f"Model saved to {checkpoint_path}")

# ===========================
# 3. 모델 로드
# ===========================
# 새로운 모델/옵티마이저 인스턴스를 만든 뒤, 저장된 state를 불러옵니다.
loaded_model = SimpleModel()
loaded_optimizer = optim.Adam(loaded_model.parameters(), lr=0.001)

checkpoint = torch.load(checkpoint_path, map_location="cpu")
loaded_model.load_state_dict(checkpoint["model_state_dict"])
loaded_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

loaded_model.eval()
print("Model loaded:")
print(loaded_model)
