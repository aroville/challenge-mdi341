import tensorflow as tf
from time import time

# Parameters
starter_learning_rate = 1e-4
training_iters = 1000000
batch_size = 2000
train_display_step = 10
valid_display_step = 50
image_size = 48
image_dim = 48*48
template_dim = 128
num_channels = 1
summaries_dir = "/home/axel/challenge-mdi341/summaries"

# Store layers weight & bias
out_channels = [1, 6, 16, 24, 32, 64, 128]
k_sizes = [7, 4, 3,  2,  2]


def get_weight_conv(i):
    k = k_sizes[i]
    in_dim = out_channels[i]
    out_dim = out_channels[i+1]
    return tf.Variable(tf.truncated_normal([k, k, in_dim, out_dim],
                                           stddev=0.05), name='wcv'+str(i+1))


def get_weight_fc(i):
    in_dim = out_channels[i] * k_sizes[i-1]**2
    out_dim = out_channels[i+1]
    return tf.Variable(tf.truncated_normal([in_dim, out_dim]), name='wfc')


def get_bias(i):
    dim = out_channels[i+1]
    return tf.Variable(tf.truncated_normal([dim]), name='b'+str(i))

w = [get_weight_conv(i) for i in range(5)]
w.append(get_weight_fc(5))
b = [get_bias(i) for i in range(6)]


def get_total_param():
    total_parameters = 0
    for variable in tf.trainable_variables():
        local_parameters = 1
        shape = variable.get_shape()
        for i in shape:
            local_parameters *= i.value
        total_parameters += local_parameters

    return total_parameters


def conv_layer(x, i, p='SAME'):
    s = [1, 1, 1, 1]
    with tf.variable_scope('convolution_layer'):
        conv = tf.nn.conv2d(input=x, filter=w[i], strides=s, padding=p) + b[i]
        return tf.nn.relu(conv, 'activation')


def pool_layer(conv):
    m = [1, 2, 2, 1]
    print(conv)
    with tf.variable_scope('pooling_layer'):
        pool = tf.nn.max_pool(value=conv, ksize=m, strides=m, padding='SAME')
        print(pool)
        return pool


def fully_connected_layer(pool, i):
    with tf.variable_scope('fully_connected_layer'):
        fc = tf.reshape(pool, [-1, int(w[i].shape[0])])
        relu = tf.nn.relu(tf.matmul(fc, w[i]) + b[i])
    print(relu)
    return relu


def predict(x):
    x = tf.reshape(x, shape=[-1, image_size, image_size, num_channels])
    l1 = pool_layer(conv_layer(x, 0))
    l2 = pool_layer(conv_layer(l1, 1))
    l3 = pool_layer(conv_layer(l2, 2))
    l4 = pool_layer(conv_layer(l3, 3, p='VALID'))
    l5 = pool_layer(conv_layer(l4, 4))
    fc = fully_connected_layer(l5, 5)
    return fc


# tf Graph input
def train(train_set, valid_set, test_set):
    x = tf.placeholder(tf.float32, [None, image_dim], name='x')
    y = tf.placeholder(tf.float32, [None, template_dim], name='y')

    # Construct model
    pred = predict(x)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.squared_difference(pred, y))
    global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(
    #     learning_rate=starter_learning_rate,
    #     global_step=global_step,
    #     decay_steps=100,
    #     decay_rate=0.8,
    #     staircase=True)
    learning_rate = 1e-4
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
        saver = tf.train.Saver()

        step = 1
        batch_valid_x, batch_valid_y = valid_set.next_batch(10000)
        t0 = time()

        # Keep training until reach max iterations
        while step < training_iters:
            batch_x, batch_y = train_set.next_batch(batch_size)
            _, summary = sess.run([optimizer, summary_op],
                                  feed_dict={x: batch_x, y: batch_y})
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
                    print("EPOCH {} valid loss: {:5.9f} ({:3.2f}s)".format(
                        train_set.epochs_completed+1, valid_loss, time() - t0))
                    print('learning rate:', learning_rate.eval())
                    saver.save(sess, 'checkpoints/model.ckpt')
                    t0 = time()
            step += 1
        print("Optimization Finished!")

        sess.run(tf.global_variables_initializer())

        y_valid_pred = sess.run(y, {x: test_set})

        with open('template_pred.bin', 'wb') as f:
            for i in range(y_valid_pred.shape[0]):
                f.write(y_valid_pred[i, :])
