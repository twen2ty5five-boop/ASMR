import torch


# 다음 코드는 PyTorch에서 텐서 생성, 기본 연산, 자동미분(autograd) 사용 방법을 익힐 때 사용합니다.

# 텐서 생성
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 기본 연산
print("Addition:", a + b)
print("Multiplication:", a * b)

# 자동 미분
x = torch.tensor(3.0, requires_grad=True)   # PyTorch에서는 Variable 대신 requires_grad=True 사용

# 순전파
y = x ** 2   # y = x^2

# 역전파 (자동 미분)
y.backward()

# dy/dx = 2x = 6
print("dy/dx:", x.grad.item())
