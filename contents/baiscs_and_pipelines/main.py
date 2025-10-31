import tensorflow as tf
from tensorflow.keras import layers, models, datasets, optimizers, losses, applications
import numpy as np

# ================================================================== #
#                         Table of Contents                          #
# ================================================================== #
# 1. Basic autodiff example 1
# 2. Basic autodiff example 2
# 3. Loading data from numpy
# 4. Input pipeline (tf.data) with CIFAR-10
# 5. Input pipeline for custom dataset (from_generator)
# 6. Pretrained model (Transfer Learning)
# 7. Save and load model
# ================================================================== #


# ================================================================== #
#                     1. Basic autodiff example 1                    #
# ================================================================== #

# Create variables.
x = tf.constant(1.0)
w = tf.Variable(2.0)
b = tf.Variable(3.0)

with tf.GradientTape() as tape:
    y = w * x + b  # y = 2 * x + 3

# Compute gradients dy/d[w,b].
grads = tape.gradient(y, [x, w, b])  
print(grads[0])  # None 
print(grads[1])  # dy/dw = x = 1.0
print(grads[2])  # dy/db = 1.0


# ================================================================== #
#                    2. Basic autodiff example 2                     #
# ================================================================== #

# Create tensors of shape (10, 3) and (10, 2).
x = tf.random.normal((10, 3))
y = tf.random.normal((10, 2))

# Build a simple linear layer.
linear = layers.Dense(2) 
optimizer = optimizers.SGD(learning_rate=0.01)
loss_fn = losses.MeanSquaredError()

# Forward + loss + backward + step
with tf.GradientTape() as tape:
    pred = linear(x, training=True)
    loss = loss_fn(y, pred)
print('loss:', float(loss))

grads = tape.gradient(loss, linear.trainable_variables)
for name, g in zip(['kernel', 'bias'], grads):
    print(f'dL/d{name}:', g)

optimizer.apply_gradients(zip(grads, linear.trainable_variables))

# One more forward to check loss change
pred2 = linear(x, training=False)
loss2 = loss_fn(y, pred2)
print('loss after 1 step optimization:', float(loss2))


# ================================================================== #
#                     3. Loading data from numpy                     #
# ================================================================== #

# Create a numpy array.
x_np = np.array([[1, 2], [3, 4]], dtype=np.float32)

# Convert numpy array to tensor.
y_tf = tf.convert_to_tensor(x_np)

# Convert tensor back to numpy.
z_np = y_tf.numpy()
# print(z_np)


# ================================================================== #
#                  4. Input pipeline (tf.data) with CIFAR-10         #
# ================================================================== #

# Download and construct CIFAR-10 dataset.
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Simple preprocessing: to float and normalize
def preprocess(img, label):
    img = tf.cast(img, tf.float32) / 255.0
    label = tf.cast(label, tf.int32)
    return img, label

batch_size = 64
train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
            .shuffle(10000)
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

# Fetch one batch (read from memory).
for images, labels in train_ds.take(1):
    print(images.shape)  # (64, 32, 32, 3)
    print(labels.shape)  # (64, 1)


# ================================================================== #
#              5. Input pipeline for custom dataset (generator)      #
# ================================================================== #

# Example dummy generator (replace with actual file loading for real use)
def data_generator(num_samples=1000, num_classes=10):
    for _ in range(num_samples):
        img = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8) 
        label = np.random.randint(0, num_classes, size=(1,), dtype=np.int32)
        yield img, label

output_signature = (
    tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8),
    tf.TensorSpec(shape=(1,), dtype=tf.int32),
)

custom_ds = (tf.data.Dataset.from_generator(data_generator, output_signature=output_signature)
             .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
             .batch(64)
             .prefetch(tf.data.AUTOTUNE))

# Iterate one batch
for images, labels in custom_ds.take(1):
    pass  # Training code should be written here.


# ================================================================== #
#                        6. Pretrained model                         #
# ================================================================== #

# Build a ResNet50 backbone with ImageNet weights.
base = applications.ResNet50(weights='imagenet', include_top=False, pooling='avg')
base.trainable = False  # freeze

# Replace the top for finetuning (example: 100 classes).
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Resizing(224, 224),
    base,
    layers.Dense(100, activation='softmax')  # 100 is an example
])

# Forward pass.
images = tf.random.uniform((64, 224, 224, 3))
outputs = model(images, training=False)
print(outputs.shape)  # (64, 100)


# ================================================================== #
#                      7. Save and load the model                    #
# ================================================================== #

# Save and load the entire model (recommended Keras format).
model.save('model.keras')
loaded = tf.keras.models.load_model('model.keras')

# Save and load only the weights.
model.save_weights('weights.ckpt')
model.load_weights('weights.ckpt')

# (Optional) quick compile & eval demo with random data
loaded.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
print('Model ready.')
