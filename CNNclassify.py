#Ignoring warnings
import os
import tensorflow as tf
import cv2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.FATAL)


def plot_conv_output(conv_img, name):
    import math
    """
        Makes plots of results of performing convolution
        :param conv_img: numpy array of rank 4
        :param name: string, name of convolutional layer
        :return: nothing, plots are saved on the disk
        """
    def get_grid_dim(x):
        """
            Transforms x into product of two integers
            :param x: int
            :return: two ints
            """
        factors = prime_powers(x)
        if len(factors) % 2 == 0:
            i = int(len(factors) / 2)
            return factors[i], factors[i - 1]
        
        i = len(factors) // 2
        return factors[i], factors[i]
    
    def prime_powers(n):
        """
        Compute the factors of a positive integer
        Algorithm from https://rosettacode.org/wiki/Factors_of_an_integer#Python
        :param n: int
        :return: set
        """
        factors = set()
        for x in range(1, int(math.sqrt(n)) + 1):
            if n % x == 0:
                factors.add(int(x))
                factors.add(int(n // x))
        return sorted(factors)
    
    import utils
    # make path to output folder
    plot_dir = os.path.join('conv_output', name)

    w_min = np.min(conv_img)
    w_max = np.max(conv_img)

    # get number of convolutional filters
    num_filters = conv_img.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))
        
    # iterate filters
    for l, ax in enumerate(axes.flat):
        # get a single image
        img = conv_img[0, :, :,  l]
        # put it on the grid
        ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
        # remove any labels from the axes
        ax.set_xticks([])
        ax.set_yticks([])
     # save figure
    plt.savefig('{}.png'.format(name), bbox_inches='tight')



def load_data(only_labels=False):
    cifar10 = tf.keras.datasets.cifar10.load_data()

    (x_train, y_train), (x_test, y_test) = cifar10
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)


    # normalizing data
    x_train = x_train/255
    x_test = x_test/255

    # subtracting mean
    # standardizing
    x_train = x_train - np.mean(x_train)
    x_test = x_test - np.mean(x_test)

    output = x_train, y_train, x_test, y_test
    if only_labels==True:
        output = x_test, y_test
    
    return output


def test(file, label_names,
         learning_rate = 1e-1, batch_size = 64, num_train_samples = 20000,
         num_test_samples = 10000, num_epochs = 35, num_classes = 10):
    # Defining Placeholders and Variables

    # x = tf.placeholder(tf.float32, [None, 32 , 32 , 3])
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    drop_out_flag = tf.placeholder(tf.bool)
    global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

#    # get number of channels in input
#    channels = x.get_shape()[3].value
#
#    # create weights tensor
#    weights = tf.Variable(tf.random_normal((5, 5, channels, 32]))
#
#    # add weights tensor to collection
#    tf.add_to_collection('conv_weights', weights)
#
#    # create bias tensor
#    bias = tf.Variable(tf.random_normal([32]))
#
#    # apply weights and biases
#    x_ = tf.reshape(x, shape=[-1, 32, 32, 3])
#
#    preactivations = tf.nn.conv2d(x_, weights, padding='SAME')
#    preactivations = tf.nn.bias_add(preactivations, bias)
#
#    # apply activation function, this is layer output
#    conv1 = tf.nn.relu(preactivations)

#    # add output to collection
#    tf.add_to_collection('conv_output', conv1)

    
    # Convolutional Layers
    conv1 = tf.layers.conv2d(inputs = x,
                         filters = 32,
                         kernel_size = [5,5],
                         padding = "same",
                         activation = tf.nn.relu,
                         name = "conv1")

    
    # add conv1 output to collection
    tf.add_to_collection('conv_output', conv1)
    
    conv1_maxpool = tf.layers.max_pooling2d(inputs = conv1, pool_size= [3,3],
                                         strides = 2, padding = 'same')
                                         
    # add conv1 output to collection
    tf.add_to_collection('conv_maxpool_output', conv1_maxpool)


    #conv layer 2
    conv2 = tf.layers.conv2d(inputs = conv1_maxpool, filters = 128,
                          kernel_size = [5,5], padding = "same",
                          activation = tf.nn.relu, name = 'conv2')
    conv2_maxpool = tf.layers.max_pooling2d(inputs = conv2, pool_size = [3,3],
                                         strides = 2, padding = 'same')


    #conv layer 3
    conv3 = tf.layers.conv2d(inputs = conv2_maxpool, filters = 256,
                          kernel_size = [5,5], padding = "same",
                          activation = tf.nn.relu, name = 'conv3')
    conv3_maxpool = tf.layers.max_pooling2d(inputs = conv3, pool_size = [3,3],
                                         strides = 2, padding = 'same')


    # Adding Fully Connected Layers

    # Flattening
    # to make compatible with fullyconnected layers
    flattened = tf.contrib.layers.flatten(conv3_maxpool)

    # Fully Connected Layer 1
    fc1 = tf.layers.dense(flattened, 2048, activation = tf.nn.relu)
    # Fully Connected Layer 2
    fc2 = tf.layers.dense(fc1, 1024, activation = tf.nn.relu)
    # Fully Connected Layer 3
    # Last Layer with Softmax
    softmax = tf.layers.dense(fc2, 10, activation = tf.nn.softmax)
    
    output_class = tf.argmax(softmax,axis=1)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1).minimize(loss, global_step=global_step)
        
    correct_prediction = tf.equal(output_class, tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    
    img = cv2.imread(file)

    if img.any():
        image = img.reshape(1, 32, 32, 3)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            
            saver.restore(sess, tf.train.latest_checkpoint("model/"))
            
            label = label_names[sess.run(tf.argmax(softmax, axis=1), feed_dict={x:image})[0]]
            
            # get output of all convolutional layers
            # here we need to provide an input image
            conv_out = sess.run([tf.get_collection('conv_output')], feed_dict={x: image})

            for i, c in enumerate(conv_out[0]):
                plot_conv_output(c, 'CONV_rslt.png')

    return label



def train(x_train, y_train, x_test, y_test,
          learning_rate = 1e-1, batch_size = 64, num_train_samples = 20000,
          num_test_samples = 10000, num_epochs = 35, num_classes = 10):

    # Defining Placeholders and Variables

    # x = tf.placeholder(tf.float32, [None, 32 , 32 , 3])
    x = tf.placeholder(tf.float32, [None, 32, 32, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    drop_out_flag = tf.placeholder(tf.bool)
    global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')


    # Convolutional Layers
    conv1 = tf.layers.conv2d(inputs = x,
                             filters = 32,
                             kernel_size = [5,5],
                             padding = "same",
                             activation = tf.nn.relu,
                             name = "conv1")

    conv1_maxpool = tf.layers.max_pooling2d(inputs = conv1, pool_size= [3,3],
                                            strides = 2, padding = 'same')


    #conv layer 2
    conv2 = tf.layers.conv2d(inputs = conv1_maxpool, filters = 128,
                             kernel_size = [5,5], padding = "same",
                             activation = tf.nn.relu, name = 'conv2')
    conv2_maxpool = tf.layers.max_pooling2d(inputs = conv2, pool_size = [3,3],
                                            strides = 2, padding = 'same')


    #conv layer 3
    conv3 = tf.layers.conv2d(inputs = conv2_maxpool, filters = 256,
                             kernel_size = [5,5], padding = "same",
                             activation = tf.nn.relu, name = 'conv3')
    conv3_maxpool = tf.layers.max_pooling2d(inputs = conv3, pool_size = [3,3],
                                            strides = 2, padding = 'same')


    # Adding Fully Connected Layers
    
    # Flattening
    # to make compatible with fullyconnected layers
    flattened = tf.contrib.layers.flatten(conv3_maxpool)

    # Fully Connected Layer 1
    fc1 = tf.layers.dense(flattened, 2048, activation = tf.nn.relu)
    # Fully Connected Layer 2
    fc2 = tf.layers.dense(fc1, 1024, activation = tf.nn.relu)
    # Fully Connected Layer 3
    # Last Layer with Softmax
    softmax = tf.layers.dense(fc2, 10, activation = tf.nn.softmax)



    output_class = tf.argmax(softmax,axis=1)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999,
                                       epsilon=1).minimize(loss, global_step=global_step)

    correct_prediction = tf.equal(output_class, tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()

    tf.global_variables_initializer().run()
    saver = tf.train.Saver(max_to_keep=None)

    #Log Accuracy
    print('\n\n\n{0:>7} {1:>12} {2:>12} {3:>12} {4:>12}'.format('Loop', 'Train Loss', 'Train Acc %', 'Test Loss', 'Test Acc %'))
    
    # Training Begins
    for i in range(num_epochs):
        epoch_loss = 0
        batch_accuracy = 0

        losses = []
        accuracies = []
        for s in range(int(num_train_samples/batch_size)):
            batch_xs = x_train[s*batch_size:(s+1)*batch_size]
            batch_ys = y_train[s*batch_size:(s+1)*batch_size]

            _, _, train_loss, train_acc = sess.run([global_step, optimizer, loss, accuracy], feed_dict ={x:batch_xs, y: batch_ys, drop_out_flag: True})
            losses.append(train_loss)
            accuracies.append(train_acc)
        avgTrainLoss = sum(losses) / len(losses)
        avgTrainAcc = sum(accuracies) / len(accuracies)

        losses = []
        accuracies = []
        for s in range(int(num_test_samples/batch_size)):
            batch_xs = x_test[s*batch_size:(s+1)*batch_size]
            batch_ys = y_test[s*batch_size:(s+1)*batch_size]

            _, _, test_loss, test_acc = sess.run([global_step, optimizer, loss, accuracy], feed_dict ={x:batch_xs, y: batch_ys,drop_out_flag: True})
            losses.append(test_loss)
            accuracies.append(test_acc)
        avgTestLoss = sum(losses) / len(losses)
        avgTestAcc = sum(accuracies) / len(accuracies)

        # Log Output
        print('{0:>7} {1:>12.4f} {2:>12.4f} {3:>12.4f} {4:>12.4f}'.format(str(i+1)+"/"+str(num_epochs), avgTrainLoss, avgTrainAcc*100, avgTestLoss, avgTestAcc*100))

    
    savePath = saver.save(sess, './model/model.ckpt')
    print('Model saved in file: {0}'.format(savePath))

def predictOutput(sess, inputX):
    return sess.run([predict],feed_dict={x: inputX})
    
def getAcc(sess, inputX, inputY):
    return sess.run([accuracy],feed_dict={x: inputX, y: inputY})



def main():
    import sys
    
    if len(sys.argv) > 0:
        # Training
        if str(sys.argv[1]) == "train":
            # Log directory
            if tf.gfile.Exists('./model'):
                tf.gfile.DeleteRecursively('./model')
            tf.gfile.MakeDirs('./model')
            
            #loading data
            x_train, y_train, x_test, y_test = load_data()
            #training model
            train(x_train, y_train, x_test, y_test, num_epochs = 16)
        
        # Testing
        if str(sys.argv[1]) == "test":
            if len(sys.argv)>2:
                file_name = sys.argv[2]
                
                label_names = ['airplane', 'automobile', 'bird', 'cat',
                               'deer', 'dog', 'frog', 'horse',
                               'ship', 'truck']
                
                #loading labels
                x_test, y_test = load_data(only_labels=True)
                #testing image
                print(test(file_name, label_names))
            
            else:
                print("Please pass the image")
    else:
        print("You forgot to specify train or test while pasing command-line arguments!")

main()
