import time
import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def batcher(X_data, y_data, batch_size=-1, random_seed=None):
    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    if random_seed is None:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(random_seed)

    rnd_idx = np.random.permutation(len(X_data))
    # print('rnd_idx[:10] is', rnd_idx[:10])
    n_batches = len(X_data) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X_data[batch_idx], y_data[batch_idx]
        yield X_batch, y_batch

n_epochs = 200
learning_rate = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape

X_data = housing.data
X_data=StandardScaler().fit_transform(X_data)
X_data = np.c_[np.ones((m, 1)), X_data]

y_data = housing.target.reshape(-1, 1)

print('X_data.shape is', X_data.shape, 'y_data.shape is', y_data.shape)

print('y_data[:30, :] is ', y_data[:30, :])

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

X = tf.placeholder(dtype=tf.float32, shape=(None, n+1), name='X')
y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

batch_size = 128

# gradients = 2/m * tf.matmul(tf.transpose(X), error)
# training_op = tf.assign(theta, theta - learning_rate * gradients)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)

alpha = 0.005
# loss = tf.add(mse, alpha*tf.nn.l2_loss(theta))
# loss = tf.add(mse, alpha*tf.nn.l2_loss(theta))
loss = tf.add(mse, tf.contrib.layers.l1_regularizer(0.02)(theta))
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    mse_val = sess.run(mse, feed_dict={X: X_val, y: y_val})
    print("first MSE =", mse_val)
    for epoch in range(n_epochs):
        # print('epoch is ', epoch)
        for X_batch, y_batch in batcher(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if epoch % 10 == 0:
            mse_val = sess.run(mse, feed_dict={X: X_val, y: y_val})
            print("Epoch", epoch, "MSE =", mse_val)
    best_theta = theta.eval()
    print('best_theta is ', best_theta)
    mse_val = sess.run(mse, feed_dict={X: X_val, y: y_val})
    print("final MSE =", mse_val)




