import tensorflow as tf
import numpy as np

# 다음코드는 선형회귀(Linear Regression)를 텐서플로우로 직접 구현하여 학습 과정(Forward → Loss → Backpropagation)의 기본 구조를 이해할 때 사용합니다.

# 데이터 준비 (y = 3x + 2)
x = np.random.rand(100).astype(np.float32)
y = 3 * x + 2 + np.random.normal(0, 0.1, 100)

W = tf.Variable(0.0)
b = tf.Variable(0.0)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

for step in range(200):
    with tf.GradientTape() as tape:
        pred = W * x + b
        loss = tf.reduce_mean((pred - y) ** 2)

    grads = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(grads, [W, b]))

    if step % 20 == 0:
        print(step, "Loss:", loss.numpy(), "W:", float(W), "b:", float(b))