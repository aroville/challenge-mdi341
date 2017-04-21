import tensorflow as tf
from time import time

# Parameters
starter_learning_rate = 0.001
training_iters = 1000000
batch_size = 1000
train_display_step = 10
valid_display_step = 100
image_size = 48
image_dim = 48*48
template_dim = 128
num_channels = 1
summaries_dir = "/home/axel/challenge-mdi341/summaries"

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.truncated_normal([5, 5, 1, 8], stddev=0.05), name='wc1'),
    'wc2': tf.Variable(tf.truncated_normal([4, 4, 8, 16], stddev=0.05), name='wc2'),
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 16, 32], stddev=0.05), name='wc3'),
    'wc4': tf.Variable(tf.truncated_normal([2, 2, 32, 64], stddev=0.05), name='wc4'),
    'wc5': tf.Variable(tf.truncated_normal([2, 2, 64, 128], stddev=0.05), name='wc5'),
    # 'wc6': tf.Variable(tf.truncated_normal([5, 5, 24, 24], stddev=0.05), name='wc6'),
    'wfc': tf.Variable(tf.truncated_normal([128, 128]), name='wfc')
}

biases = {
    'bc1': tf.Variable(tf.truncated_normal([8]), name='bc1'),
    'bc2': tf.Variable(tf.truncated_normal([16]), name='bc2'),
    'bc3': tf.Variable(tf.truncated_normal([32]), name='bc3'),
    'bc4': tf.Variable(tf.truncated_normal([64]), name='bc4'),
    'bc5': tf.Variable(tf.truncated_normal([128]), name='bc5'),
    # 'bc6': tf.Variable(tf.truncated_normal([24]), name='bc6'),
    'bfc': tf.Variable(tf.truncated_normal([128]), name='bfc')
}


def get_total_param():
    total_parameters = 0
    for variable in tf.trainable_variables():
        local_parameters = 1
        shape = variable.get_shape()
        for i in shape:
            local_parameters *= i.value
        total_parameters += local_parameters

    return total_parameters


def conv_layer(x, w_key, b_key, k=2):
    with tf.variable_scope('convolution_layer'):
        conv = tf.nn.conv2d(input=x,
                            filter=weights[w_key],
                            strides=[1, k, k, 1],
                            padding='SAME') + biases[b_key]
        relu = tf.nn.relu(conv, 'activation')
    print(relu)
    return relu


def pool_layer(conv, k=2):
    with tf.variable_scope('pooling_layer'):
        pool = tf.nn.max_pool(value=conv,
                              ksize=[1, k, k, 1],
                              strides=[1, k, k, 1],
                              padding='SAME')
    print(pool)
    return pool


def fully_connected_layer(pool, size, w_key, b_key):
    with tf.variable_scope('fully_connected_layer'):
        fc = tf.reshape(pool, [-1, size])
        fc = tf.matmul(fc, weights[w_key]) + biases[b_key]
        relu = tf.nn.relu(fc)
    print(relu)
    return relu


def predict(x):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, image_size, image_size, num_channels])

    conv1 = conv_layer(x, 'wc1', 'bc1')
    pool1 = pool_layer(conv1)
    conv2 = conv_layer(pool1, 'wc2', 'bc2')
    pool2 = pool_layer(conv2)
    conv3 = conv_layer(pool2, 'wc3', 'bc3')
    pool3 = pool_layer(conv3)
    conv4 = conv_layer(pool3, 'wc4', 'bc4')
    pool4 = pool_layer(conv4)
    conv5 = conv_layer(pool4, 'wc5', 'bc5')
    pool5 = pool_layer(conv5)

    return fully_connected_layer(pool5, 128, 'wfc', 'bfc')


# tf Graph input
def train(train_set, valid_set, test_set):
    x = tf.placeholder(tf.float32, [None, image_dim], name='x')
    y = tf.placeholder(tf.float32, [None, template_dim], name='y')

    # Construct model
    pred = predict(x)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.squared_difference(pred, y))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate=starter_learning_rate,
        global_step=global_step,
        decay_steps=5000,
        decay_rate=0.96,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
        .minimize(cost, global_step=global_step)

    summary_op = tf.summary.merge_all()

    # Initializing the variables
    init = tf.global_variables_initializer()
    print('Nb of parameters:', get_total_param())

    # Launch the graph
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(summaries_dir, sess.graph)

        sess.run(init)
        step = 1
        batch_valid_x, batch_valid_y = valid_set.next_batch(10000)
        t0 = time()

        # Keep training until reach max iterations
        while step < training_iters:
            batch_x, batch_y = train_set.next_batch(batch_size)
            _, summary = sess.run([optimizer, summary_op], feed_dict={x: batch_x, y: batch_y})
            writer.add_summary(summary, step)

            if step % train_display_step == 0:
                train_loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                tf.summary.scalar('train_loss', train_loss)

                print("nb images seen: {:8}".format(step * batch_size),
                      "\t\tloss: {:5.9f}".format(train_loss))

                if step % valid_display_step == 0:
                    valid_loss = sess.run(cost, feed_dict={
                        x: batch_valid_x,
                        y: batch_valid_y
                    })
                    tf.summary.scalar('valid_loss', valid_loss)
                    print("---------- valid loss: {:5.9f} ({:3.2f}s)".format(valid_loss, time() - t0))
                    t0 = time()
            step += 1
        print("Optimization Finished!")

        # Calculate accuracy for 256 mnist test images
        print("Testing Accuracy:",
              sess.run(cost, feed_dict={x: valid_set.images,
                                        y: valid_set.labels}))

        saver = tf.train.Saver()
        saver.save(sess, './model.ckpt')

        sess.run(tf.global_variables_initializer())

        y_valid_pred = sess.run(y, {x: test_set})

        with open('template_pred.bin', 'wb') as f:
            for i in range(y_valid_pred.shape[0]):
                f.write(y_valid_pred[i, :])
