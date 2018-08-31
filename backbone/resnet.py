import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data


def residual_block(net, block, repeat, name, use_stride=True, is_training=None):
    print("block_%s" % name)
    for i in range(repeat):
        short_cut = net
        for j, filter in enumerate(block):
            stride = 1
            if i == 0 and j == 0 and use_stride:
                stride = 2
            net = tf.layers.conv2d(net, filter[1], filter[0], stride, 'same', name="%s_%d_%d" % (name, i, j),
                                   use_bias=False)
            net = tf.layers.batch_normalization(net, training=is_training)
            print(net)
            if j > len(block) - 1:
                net = tf.nn.relu(net, name="%s_relu_%d_%d" % (name, i, j))
        short_cut_channel = short_cut.get_shape()[3]
        last_layer_channel = net.get_shape()[3]

        stride = 1
        if i == 0 and use_stride:
            stride = 2

        if short_cut_channel == last_layer_channel:
            if stride > 1:
                short_cut = tf.layers.max_pooling2d(short_cut, 1, strides=stride)
        else:
            short_cut = tf.layers.conv2d(short_cut, int(net.get_shape()[3]), 1, stride, 'same',
                                         name="%s_projection_%d_%d" % (name, i, j))
        net += short_cut
        net = tf.nn.relu(net, name="%s_relu_%d_%d" % (name, i, j))
    return net


def resnet_18(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[3, 64], [3, 64]], 2, "conv2", False, is_training=is_training)
    net = residual_block(net, [[3, 128], [3, 128]], 2, "conv3", True, is_training)
    net = residual_block(net, [[3, 256], [3, 256]], 2, "conv4", True, is_training)
    net = residual_block(net, [[3, 512], [3, 512]], 2, "conv5", True, is_training)
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


def resnet_34(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[3, 64], [3, 64]], 3, "conv2", False, is_training=is_training)
    net = residual_block(net, [[3, 128], [3, 128]], 4, "conv3", True, is_training)
    net = residual_block(net, [[3, 256], [3, 256]], 6, "conv4", True, is_training)
    net = residual_block(net, [[3, 512], [3, 512]], 3, "conv5", True, is_training)
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


def resnet_50(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    net = residual_block(net, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    net = residual_block(net, [[1, 256], [3, 256], [1, 1024]], 6, "conv4", True, is_training)
    net = residual_block(net, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


def resnet_101(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    net = residual_block(net, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    net = residual_block(net, [[1, 256], [3, 256], [1, 1024]], 23, "conv4", True, is_training)
    net = residual_block(net, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


def resnet_152(input, num_class=10, is_training=None):
    net = tf.layers.conv2d(input, 64, 7, 2, 'same', name="conv1", use_bias=False)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, 3, 2, padding="same", name="pool1")
    net = residual_block(net, [[1, 64], [3, 64], [1, 256]], 3, "conv2", False, is_training=is_training)
    net = residual_block(net, [[1, 128], [3, 128], [1, 512]], 4, "conv3", True, is_training)
    net = residual_block(net, [[1, 256], [3, 256], [1, 1024]], 36, "conv4", True, is_training)
    net = residual_block(net, [[1, 512], [3, 512], [1, 2048]], 3, "conv5", True, is_training)
    net = tf.reduce_mean(net, [1, 2], name='last_pool')
    net = tf.layers.dense(net, num_class, name='logits')
    return net


num_class = 10
batch_size = 32
image_size = 28
input_size = 224
num_channel = 1
learning_rate = 0.01
num_iter = 1000

inputs = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_channel], name="input")
labels = tf.placeholder(tf.int32, shape=[None, num_class], name="label")
is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

logits = resnet_18(tf.image.resize_images(inputs, (input_size, input_size)), num_class, is_training)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
predict = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(labels, 1)), tf.float32))
train = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1.0).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

test_path = os.path.dirname(os.path.realpath(__file__))
if not os.path.exists(os.path.join(test_path, 'mnist')):
    os.makedirs(os.path.join(test_path, 'mnist'))
data_sets = input_data.read_data_sets(test_path, one_hot=True)

for step in range(num_iter):
    train_x, train_y = data_sets.train.next_batch(batch_size)
    train_x = train_x.reshape((batch_size, 28, 28, 1))
    _, accuracy_result, loss_result = sess.run([train, accuracy, loss], feed_dict={inputs: train_x, labels: train_y,
                                                                                   is_training: True})
    print('%d train accuracy: %f, loss: %f' % (step, accuracy_result, loss_result))

    if step % 100 == 0 and step > 0:
        valid_x, valid_y = data_sets.validation.next_batch(batch_size)
        valid_x = valid_x.reshape((batch_size, 28, 28, 1))
        accuracy_result, loss_result = sess.run([accuracy, loss],
                                                feed_dict={inputs: valid_x, labels: valid_y,
                                                           is_training: True})
        print('%d validation accuracy: %f, loss: %f' % (step, accuracy_result, loss_result))
