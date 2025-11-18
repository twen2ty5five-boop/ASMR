import torch
import numpy as np

# 다음 코드는 선형회귀(Linear Regression)를 PyTorch로 직접 구현하여
# 학습 과정(Forward → Loss → Backpropagation → Update)의 기본 구조를 이해할 때 사용합니다.

# 데이터 준비 (y = 3x + 2)
x = np.random.rand(100).astype(np.float32)
y = 3 * x + 2 + np.random.normal(0, 0.1, 100)

# numpy → torch 텐서 변환
x_t = torch.from_numpy(x)              # (100,)
y_t = torch.from_numpy(y)              # (100,)

# 학습할 파라미터 W, b (초기값 0.0)
W = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

optimizer = torch.optim.SGD([W, b], lr=0.1)

for step in range(200):
    # ----- Forward -----
    pred = W * x_t + b          # 예측값
    loss = torch.mean((pred - y_t) ** 2)  # MSE 손실

    # ----- Backward -----
    optimizer.zero_grad()       # 기존 gradient 초기화
    loss.backward()             # dL/dW, dL/db 계산

    # ----- Update -----
    optimizer.step()            # W, b 갱신

    if step % 20 == 0:
        print(step,
              "Loss:", loss.item(),
              "W:", W.item(),
              "b:", b.item())
