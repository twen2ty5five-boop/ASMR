import tensorflow as tf

# 다음 코드는 텐서플로우의 기초 연산, 텐서 생성, 자동미분(GradientTape) 사용 방법을 익힐 때 사용합니다.

# 텐서 생성
a = tf.constant([1.0, 2.0, 3.0])
b = tf.constant([4.0, 5.0, 6.0])

# 기본 연산
print("Addition:", a + b)
print("Multiplication:", a * b)

# 자동 미분
x = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = x ** 2  # y = x^2

grad = tape.gradient(y, x)
print("dy/dx:", grad)