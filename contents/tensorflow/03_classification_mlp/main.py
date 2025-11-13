import tensorflow as tf

# 다음은 다층 퍼셉트론(MLP)으로 기본 분류(Classification) 모델을 만들 때 사용합니다. MNIST 숫자 분류의 가장 기본 구조입니다.

# MNIST 데이터 로드
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784) / 255.0
x_test  = x_test.reshape(-1, 784) / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
model.evaluate(x_test, y_test)