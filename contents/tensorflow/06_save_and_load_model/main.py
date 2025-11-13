import tensorflow as tf

# 다음은 학습한 모델을 저장하고, 다시 로드하여 사용하는 기본 구조입니다.

# 간단한 모델
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.save("saved_model")

# --- 불러오기 ---
loaded = tf.keras.models.load_model("saved_model")
print("Model loaded:", loaded.summary())