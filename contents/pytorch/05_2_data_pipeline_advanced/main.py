import torch
from torch.utils.data import Dataset, DataLoader

# 다음 코드는 PyTorch에서 Dataset + DataLoader를 사용하여
# 대규모 데이터 파이프라인을 효율적으로 구성할 때 사용합니다.
# 특히 num_workers, prefetch_factor를 이용해 병렬 로딩 및 프리페치 개념을 보여줍니다.

# ============================
# 1. 가짜 데이터 생성
# ============================
x = torch.arange(10)   # [0,1,2,...,9]
y = x * 2              # 라벨은 x*2

# (x, y) 텐서를 Dataset 형태로 묶기 위한 커스텀 Dataset
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
# - shuffle=True  : 무작위 섞기
# - batch_size=3  : 배치 크기 3
# - num_workers=2 : 데이터 로딩을 2개의 워커 프로세스로 병렬 처리
# - prefetch_factor=2 : 각 워커가 미리 2배치씩 준비해 두도록 프리페치
data_loader = DataLoader(
    dataset,
    batch_size=3,
    shuffle=True,
    num_workers=2,
    prefetch_factor=2,
    persistent_workers=True,  # 여러 epoch에 걸쳐 워커 유지 (옵션)
)

# ============================
# 3. 배치 단위로 데이터 가져오기
# ============================
for batch_x, batch_y in data_loader:
    print(batch_x.numpy(), batch_y.numpy())



[결과 해석 관련]
5-1: DataLoader의 기본 shuffle + batch 동작 확인용 예제


5-2: 동일한 데이터지만 num_workers, prefetch_factor를 설정해
 병렬 로딩 + 프리페치를 적용한 예제이고,
 출력 순서가 달라질 수 있으나 y = 2x 관계는 유지됨
