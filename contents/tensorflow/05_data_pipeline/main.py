import tensorflow as tf

# 다음은 tf.data.Dataset으로 대규모 데이터 파이프라인을 효율적으로 구성할 때 사용합니다.

# 가짜 데이터 생성
x = tf.range(10)
y = x * 2

dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.shuffle(10).batch(3)

for batch_x, batch_y in dataset:
    print(batch_x.numpy(), batch_y.numpy())