import numpy
import tensorflow as tf


n_nodes_hl1 = 2
n_nodes_hl2 = 2
n_nodes_hl3 = 2
n_classes = 1

numpy.float32

input_x = [numpy.float32(0.0),numpy.float32(1.0)],[numpy.float32(0.0),numpy.float32(0.0)]
target = [numpy.float32(1.0)],[numpy.float32(0.0)]
x = tf.placeholder('float', [None, 2])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([2, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.matmul(data,hidden_1_layer['weights']) + hidden_1_layer['biases']
    l1 = tf.nn.relu(l1)

    l2 = tf.matmul(l1,hidden_2_layer['weights']) + hidden_2_layer['biases']
    l2 = tf.nn.relu(l2)

    l3 = tf.matmul(l2,hidden_3_layer['weights']) +  hidden_3_layer['biases']
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x_input,y_target):
    x_input = numpy.array(x_input).reshape(1,2,len(x_input))
    y_target = numpy.array(y_target)
    prediction = neural_network_model(x_input[0])
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y_target))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        # sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(0,len(x_input)):
                epoch_x, epoch_y = x_input[_],y_target[_]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: input_x, y: target}))




train_neural_network(input_x,target)
