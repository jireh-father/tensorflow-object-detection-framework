import tensorflow as tf
from data.pascal_voc import pascal_voc
from model import yolo

max_iter = 10000
pascal = pascal_voc('train')
inputs = tf.placeholder(tf.float32, [None, 448, 448, 3])
labels = tf.placeholder(tf.float32, [None, 7, 7, 5 + 20])

is_training = tf.placeholder(tf.bool, shape=(), name="is_training")

logits = yolo.build_model(inputs, is_training)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
predict = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, tf.argmax(labels, 1)), tf.float32))
train = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1.0).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(1, max_iter):
    tr_x, tr_y = pascal.get()
    feed_dict = {inputs: tr_x, labels: tr_y, is_training: True}
    _, accuracy_result, loss_result = sess.run([train, accuracy, loss], feed_dict={inputs: tr_x, labels: tr_y,
                                                                                   is_training: True})
    print('%d train accuracy: %f, loss: %f' % (step, accuracy_result, loss_result))

    if step % 100 == 0 and step > 0:
        valid_x, valid_y = data_sets.validation.next_batch(batch_size)
        valid_x = valid_x.reshape((batch_size, 28, 28, 1))
        accuracy_result, loss_result = sess.run([accuracy, loss],
                                                feed_dict={inputs: valid_x, labels: valid_y,
                                                           is_training: True})
        print('%d validation accuracy: %f, loss: %f' % (step, accuracy_result, loss_result))

for step in range(1, max_iter + 1):
    tr_x, tr_y = pascal.get()
    feed_dict = {inputs: tr_x, labels: tr_y}

    summary_str, loss, _ = self.sess.run(
        [self.summary_op, self.net.total_loss, self.train_op],
        feed_dict=feed_dict)
    train_timer.toc()

    log_str = '''{} Epoch: {}, Step: {}, Learning rate: {},'''
    ''' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'''
    '''' Load: {:.3f}s/iter, Remain: {}'''.format(
        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
        self.data.epoch,
        int(step),
        round(self.learning_rate.eval(session=self.sess), 6),
        loss,
        train_timer.average_time,
        load_timer.average_time,
        train_timer.remain(step, self.max_iter))
    print(log_str)

    self.writer.add_summary(summary_str, step)
