import tensorflow as tf
print("TensorFlow version:", tf.__version__)

# +
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# +
print(x_train.shape, x_train.dtype)
print(y_train.shape, y_train.dtype)
print(x_test.shape, x_test.dtype)
print(y_test.shape, y_test.dtype)
import matplotlib.pyplot as plt

plt.imshow(x_train[:1][0])

# +
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# As I understand, the first one is the first layer,
# where we flatten the 28x28 input.
# the second is one of the hidden layers with relu
# activation, with 128 neurons
# the dropout part, i didn't get it.
# the last dense part is the output, giving something between
# 0-9, as this is a number recognition problem.

# +
predictions = model(x_train[:1]).numpy()
predictions

# first, we feed in the test data and get some predictions
# as we haven't trained the network yet, the result would be
# rubbish.
# -

tf.nn.softmax(predictions).numpy()
# I think this normalizes the output to 1
# in a nonlinear way?

# +
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# let's see how the loss functions is designed.

# +
loss_fn(y_train[:1], predictions).numpy()

# compares predictions vs the actual labels (desired outputs)
# and calculates the loss function. The resulting value is
# to be minimized.
# -

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# +
model.fit(x_train, y_train, epochs=5)

# I believe this is the actual training,
# -

model.evaluate(x_test,  y_test, verbose=2)


probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])


