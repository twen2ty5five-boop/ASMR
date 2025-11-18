import torch
from torch.utils.data import Dataset, DataLoader

# 다음 코드는 PyTorch에서 Dataset + DataLoader를 사용하여
# 대규모 데이터 파이프라인을 효율적으로 구성할 때 사용합니다.
# 아래 예시는 텐서플로우의 tf.data.Dataset.from_tensor_slices와 동일하게
# (x, y) 데이터를 묶어서 배치 단위로 로드하는 기본 구조입니다.

# ============================
# 1. 가짜 데이터 생성
# ============================
x = torch.arange(10)           # [0,1,2,...,9]
y = x * 2                      # 라벨은 x*2

# 텐서 두 개를 Dataset 형태로 묶기 위한 커스텀 Dataset 정의
class TensorPairDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset = TensorPairDataset(x, y)

# ============================
# 2. DataLoader로 파이프라인 구성
# ============================
# TF의 dataset.shuffle(10).batch(3) 와 동일하게:
data_loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True
)

# ============================
# 3. 배치 단위로 데이터 가져오기
# ============================
for batch_x, batch_y in data_loader:
    print(batch_x.numpy(), batch_y.numpy())
