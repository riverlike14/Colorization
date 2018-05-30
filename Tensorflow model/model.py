import tensorflow as tf
import numpy as np
from skimage.color import lab2rgb
from skimage.transform import pyramid_expand as upscale
from time import time

class model():
    def __init__(self):
        self.Q = 266
        self.T = 0.38
        
        self.X = None
        self.y = None
        self.weight = None
        self.prediction = None
        self.train_step = None
        self.cost = None
    
    def build_model(self, shape):
        N, H, W, _ = shape
        X = tf.placeholder(tf.float32, [N, H, W, 1], name='X')
        y = tf.placeholder(tf.float32, [N, H//4, W//4, self.Q], name='y')
        weight = tf.placeholder(tf.float32, [N, H//4, W//4], name="weight")

        def weight_variable(shape, name=None):
            initial = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(initial, name=name)

        def bias_variable(shape, name=None):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial, name=name)

        def conv2d(x, W, stride=1, dilation=1, name=None):
            strides = [1, stride, stride, 1]
            dilations = [1, 1, dilation, dilation]
            return tf.nn.conv2d(x, W, padding="SAME", strides=strides, dilations=dilations, name=name)

        def conv2d_T(x, W, output_shape, stride=1, name=None):
            strides = [1, stride, stride, 1]
            # output_shape = [N, 2*H, 2*W, C]
            return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=strides, name=name)

        def batch_normalize(x, shape, epsilon=1e-3, name=None):
            batch_mean, batch_var = tf.nn.moments(x, [0])
            beta = tf.Variable(tf.zeros(shape))
            scale = tf.Variable(tf.ones(shape))
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon, name=name)

        with tf.name_scope("conv1"):
            W_conv1_1 = weight_variable([3, 3, 1, 64])
            b_conv1_1 = bias_variable([64])
            conv1_1 = conv2d(X, W_conv1_1) + b_conv1_1
            conv1_1 = tf.nn.relu(conv1_1)

            W_conv1_2 = weight_variable([3, 3, 64, 64])
            b_conv1_2 = bias_variable([64])
            conv1_2 = conv2d(conv1_1, W_conv1_2, stride=2) + b_conv1_2 # Halve size (1/2)
            conv1_2 = tf.nn.relu(conv1_2)

            conv1_norm = batch_normalize(conv1_2, [64])                                 

        with tf.name_scope("conv2"):
            W_conv2_1 = weight_variable([3, 3, 64, 128])
            b_conv2_1 = bias_variable([128])
            conv2_1 = conv2d(conv1_norm, W_conv2_1) + b_conv2_1
            conv2_1 = tf.nn.relu(conv2_1)

            W_conv2_2 = weight_variable([3, 3, 128, 128])
            b_conv2_2 = bias_variable([128])
            conv2_2 = conv2d(conv2_1, W_conv2_2, stride=2) + b_conv2_2 # Halve size (1/4)
            conv2_2 = tf.nn.relu(conv2_2)

            conv2_norm = batch_normalize(conv2_2, [128])

        with tf.name_scope("conv3"):
            W_conv3_1 = weight_variable([3, 3, 128, 256])
            b_conv3_1 = bias_variable([256])
            conv3_1 = conv2d(conv2_norm, W_conv3_1) + b_conv3_1
            conv3_1 = tf.nn.relu(conv3_1)

            W_conv3_2 = weight_variable([3, 3, 256, 256])
            b_conv3_2 = bias_variable([256])
            conv3_2 = conv2d(conv3_1, W_conv3_2) + b_conv3_2
            conv3_2 = tf.nn.relu(conv3_2)

            W_conv3_3 = weight_variable([3, 3, 256, 256])
            b_conv3_3 = bias_variable([256])
            conv3_3 = conv2d(conv3_2, W_conv3_3, stride=2) + b_conv3_3 # Halve size (1/8)
            conv3_3 = tf.nn.relu(conv3_3)

            conv3_norm = batch_normalize(conv3_3, [256])

        with tf.name_scope("conv4"):
            W_conv4_1 = weight_variable([3, 3, 256, 512])
            b_conv4_1 = bias_variable([512])
            conv4_1 = conv2d(conv3_norm, W_conv4_1) + b_conv4_1
            conv4_1 = tf.nn.relu(conv4_1)

            W_conv4_2 = weight_variable([3, 3, 512, 512])
            b_conv4_2 = bias_variable([512])
            conv4_2 = conv2d(conv4_1, W_conv4_2) + b_conv4_2
            conv4_2 = tf.nn.relu(conv4_2)

            W_conv4_3 = weight_variable([3, 3, 512, 512])
            b_conv4_3 = bias_variable([512])
            conv4_3 = conv2d(conv4_2, W_conv4_3) + b_conv4_3
            conv4_3 = tf.nn.relu(conv4_3)

            conv4_norm = batch_normalize(conv4_3, [512])

        with tf.name_scope("conv5"):
            W_conv5_1 = weight_variable([3, 3, 512, 512])
            b_conv5_1 = bias_variable([512])
            conv5_1 = conv2d(conv4_norm, W_conv5_1, dilation=2) + b_conv5_1
            conv5_1 = tf.nn.relu(conv5_1)

            W_conv5_2 = weight_variable([3, 3, 512, 512])
            b_conv5_2 = bias_variable([512])
            conv5_2 = conv2d(conv5_1, W_conv5_2, dilation=2) + b_conv5_2
            conv5_2 = tf.nn.relu(conv5_2)

            W_conv5_3 = weight_variable([3, 3, 512, 512])
            b_conv5_3 = bias_variable([512])
            conv5_3 = conv2d(conv5_2, W_conv5_3, dilation=2) + b_conv5_3
            conv5_3 = tf.nn.relu(conv5_3)

            conv5_norm = batch_normalize(conv5_3, [512])

        with tf.name_scope("conv6"):
            W_conv6_1 = weight_variable([3, 3, 512, 512])
            b_conv6_1 = bias_variable([512])
            conv6_1 = conv2d(conv5_norm, W_conv6_1, dilation=2) + b_conv6_1
            conv6_1 = tf.nn.relu(conv6_1)

            W_conv6_2 = weight_variable([3, 3, 512, 512])
            b_conv6_2 = bias_variable([512])
            conv6_2 = conv2d(conv6_1, W_conv6_2, dilation=2) + b_conv6_2
            conv6_2 = tf.nn.relu(conv6_2)

            W_conv6_3 = weight_variable([3, 3, 512, 512])
            b_conv6_3 = bias_variable([512])
            conv6_3 = conv2d(conv6_2, W_conv6_3, dilation=2) + b_conv6_3
            conv6_3 = tf.nn.relu(conv6_3)

            conv6_norm = batch_normalize(conv6_3, [512])

        with tf.name_scope("conv7"):
            W_conv7_1 = weight_variable([3, 3, 512, 512])
            b_conv7_1 = bias_variable([512])
            conv7_1 = conv2d(conv6_norm, W_conv7_1) + b_conv7_1
            conv7_1 = tf.nn.relu(conv7_1)

            W_conv7_2 = weight_variable([3, 3, 512, 512])
            b_conv7_2 = bias_variable([512])
            conv7_2 = conv2d(conv7_1, W_conv7_2) + b_conv7_2
            conv7_2 = tf.nn.relu(conv7_2)

            W_conv7_3 = weight_variable([3, 3, 512, 512])
            b_conv7_3 = bias_variable([512])
            conv7_3 = conv2d(conv7_2, W_conv7_3) + b_conv7_3
            conv7_3 = tf.nn.relu(conv7_3)

            conv7_norm = batch_normalize(conv7_3, [512])

        with tf.name_scope("deconv8"):
            W_conv8_1 = weight_variable([4, 4, 256, 512])
            b_conv8_1 = bias_variable([256])
            conv8_1 = conv2d_T(conv7_norm, W_conv8_1, output_shape=[N, H//4, W//4, 256], stride=2) + b_conv8_1 # Double size (1/4)
            conv8_1 = tf.nn.relu(conv8_1)

            W_conv8_2 = weight_variable([3, 3, 256, 256])
            b_conv8_2 = bias_variable([256])
            conv8_2 = conv2d(conv8_1, W_conv8_2) + b_conv8_2
            conv8_2 = tf.nn.relu(conv8_2)

            W_conv8_3 = weight_variable([3, 3, 256, 256])
            b_conv8_3 = bias_variable([256])
            conv8_3 = conv2d(conv8_2, W_conv8_3) + b_conv8_3
            conv8_3 = tf.nn.relu(conv8_3)

        with tf.name_scope("Softmax"):
            W_conv8_Q = weight_variable([1, 1, 256, self.Q])
            b_conv8_Q = bias_variable([self.Q])
            conv8_Q = conv2d(conv8_3, W_conv8_Q) + b_conv8_Q

            class8_Q_rh = tf.nn.softmax(conv8_Q / self.T, name="Prediction")

        with tf.name_scope("Loss"):
            cross_entropy = - tf.reduce_sum(y * tf.log(class8_Q_rh + 1e-8), axis=3)
            cost = tf.reduce_sum(weight * cross_entropy, name="Cost")
            train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
        
        self.X = X
        self.y = y
        self.weight = weight
        self.prediction = class8_Q_rh
        self.train_step = train_step
        self.cost = cost
        pass
    
    def train(self, sess, X_train, y_train, weight_train, iter_num=1000, restore_path=None, save_path=None, verbose=True):
        
        if len(X_train.shape) != 4:
            raise ValueError("Dimension of the input image must be 4.")
#         self.build_model(X_train.shape)
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        if restore_path:
            saver.restore(sess, tf.train.latest_checkpoint(restore_path))
        
        tic = time()
        for i in range(iter_num):
            if i % 100 == 0:
                toc = time()
                print("Iter: %5d / Took %7.2f seconds. / Loss: %f" % \
                      (i, toc - tic, sess.run(self.cost, feed_dict={self.X: X_train, self.y: y_train, self.weight: weight_train})))
                
            sess.run(self.train_step, feed_dict={self.X: X_train, self.y: y_train, self.weight: weight_train})

        toc = time()
        print("Took %.4f seconds." % (toc - tic))

        if save_path:
            saver.save(sess, save_path)
    
    def predict(self, sess, X_test, restore_path=None, get_image=False):        
        if len(X_test.shape) != 4:
            raise ValueError("Dimension of the input image must be 4.")
#         self.build_model(X_test.shape)
        
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver()
        if restore_path:
            saver.restore(sess, tf.train.latest_checkpoint(restore_path))

        prediction = sess.run(self.prediction, feed_dict={self.X: X_test})
        
        if get_image:
            filter_ab = np.load("Model_data/filter_ab.npy")
            color_space = np.load("Model_data/color_space.npy")

            N, H, W, _ = prediction.shape
            annealed_mean = np.sum(np.expand_dims(prediction, axis=4) * np.tile(color_space[filter_ab], (N, H, W, 1, 1)), axis=3)
            
            image_lab = np.zeros((N, H, W, 3))
            image_lab[:, :, :, 0] = X_test[:, 2:-1:4, :, 0][:, :, 2:-1:4]
            image_lab[:, :, :, 1:] = annealed_mean
            
            image_rgb = np.zeros((*X_test.shape[:3], 3), dtype=np.uint8)
            for i in range(N):
                image_rgb[i] = (255*lab2rgb(upscale(image_lab[i], 4))).astype(np.uint8)
            
            return image_rgb
        else:
            return prediction