# Multi-structure Regions of Interest
# 
# References : 
#       CNN structure based on VGG16, https://github.com/ry/tensorflow-vgg16/blob/master/vgg16.py
#       Channel independent feature maps (3D features) using https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#depthwise_conv2d_native 
#       GAP based on https://github.com/jazzsaxmafia/Weakly_detector/blob/master/src/detector.py
#       Conv2d layer based on https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py

#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import _pickle as cPickle
import utils_net as utils
from params import CNNParams, HyperParams

hyper     = HyperParams(verbose=False)
cnn_param = CNNParams(verbose=False)

def print_model_params(verbose=True):
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        if verbose: print("name: " + str(variable.name) + " - shape:" + str(shape))
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        if verbose: print("variable parameters: " , variable_parametes)
        total_parameters += variable_parametes
    if verbose: print("total params: ", total_parameters)
    return total_parameters

class CNN():
    def __init__(self, global_step):
        self._global_step = global_step
        self._counted_scope = []
        self._flops = 0
        self._weights = 0
        self.is_train = tf.placeholder(tf.bool, name="is_train")

    def load_vgg_weights(self):
        with open(hyper.vgg_weights) as f:
            self.pretrained_weights = cPickle.load(f)

    def get_vgg_weights(self, layer_name, bias=False):
        layer = self.pretrained_weights[layer_name]
        if bias: return layer[1]
        # tranpose because VGG weights were stored in diffeerent order
        return layer[0].transpose((2,3,1,0)) 

    def conv2d_depth_or_not(self, input_, name, nonlinearity=None):
        with tf.variable_scope(name) as scope:
            
            W_shape = cnn_param.layer_shapes[name + '/W']
            b_shape = cnn_param.layer_shapes[name + '/b']
            
            if hyper.fine_tuning and name not in ['conv6', 'conv6_1', 'depth']:
                # because conv6, conv6_1, and depth are the layers added on top of VGG 
                # hence not present in VGG 
                W = self.get_vgg_weights(name)
                b = self.get_vgg_weights(name, bias=True)
                W_initializer = tf.constant_initializer(W)
                b_initializer = tf.constant_initializer(b)
            else:
                W_initializer = tf.truncated_normal_initializer(stddev=hyper.stddev)
                b_initializer = tf.constant_initializer(0.0)
                
            conv_weights = tf.get_variable("W", shape=W_shape, initializer=W_initializer)
            conv_biases  = tf.get_variable("b", shape=b_shape, initializer=b_initializer)

            if name == 'depth':
                # learn different filter for each input channel
                # thus the number of input channel has to be reduced
                conv = tf.nn.depthwise_conv2d_native(input_, conv_weights, [1,1,1,1], padding='SAME')
                # conv = tf.nn.separable_conv2d(input_, conv_weights, [1,1,1,1], padding='SAME')
            else:
                conv = tf.nn.conv2d(input_, conv_weights, [1,1,1,1], padding='SAME')

            bias = tf.nn.bias_add(conv, conv_biases)
            bias = tf.nn.dropout(bias,0.7) 
            if nonlinearity is None: 
                return bias
            return nonlinearity(bias, name=name)

    # currently not required, but for experimentation purposes
    # there are two FCL layer at the end of VGG NET
    def fully_connected_layer(self, input_, input_size, output_size, name, nonlinearity=None):
        shape = input_.get_shape().to_list()
        x = tf.reshape(input_, [-1, np.prod(shape[1:])])
        with tf.variable_scope(name) as scope:
            W   = tf.get_variable("W", shape=[input_size, output_size], 
                  initializer=tf.random_normal_initializer(stddev=hyper.stddev))
            b   = tf.get_variable("b", shape=[output_size], initializer=tf.constant_initializer(0.))
            bias = tf.nn.bias_add(tf.matmul(x, W), b, name=scope)
            if nonlinearity is None: 
                return bias
            return nonlinearity(bias, name=name)
        return nonlinearity(bias, name=name)
       
    def image_conversion_scaling(self, image):
        # Conversion to bgr and mean substraction is common with VGGNET
        # Because pre-trained values use them, https://arxiv.org/pdf/1409.1556.pdf
        image *= 255.
        r, g, b = tf.split(image, 3, 3)
        VGG_MEAN = [103.939, 116.779, 123.68]
        return tf.concat([b-VGG_MEAN[0], g-VGG_MEAN[1], r-VGG_MEAN[2]], 3)


    def build(self, image):

        image = self.image_conversion_scaling(image)

        print('Building model')
        #filters = [128, 128, 256, 512, 1024]
        filters = [64, 64, 128, 256, 512]
        kernels = [7, 3, 3, 3, 3]
        strides = [2, 0, 2, 2, 2]

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            x = self._conv(image, kernels[0], filters[0], strides[0])
            x = self._bn(x)
            x = self._relu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = self._residual_block(x, name='conv2_1')
        x = self._residual_block(x, name='conv2_2')

        # conv3_x
        x = self._residual_block_first(x, filters[2], strides[2], name='conv3_1')
        x = self._residual_block(x, name='conv3_2')

        # conv4_x
        x = self._residual_block_first(x, filters[3], strides[3], name='conv4_1')
        x = self._residual_block(x, name='conv4_2')

        # conv5_x
        x = self._residual_block_first(x, filters[4], strides[4], name='conv5_1')
        x = self._residual_block(x, name='conv5_2')

        gap = tf.reduce_mean(x, [1, 2])

        # with tf.variable_scope("GAP"):
        #     gap_w = tf.get_variable("W", shape=cnn_param.layer_shapes['GAP/W'],
        #             initializer=tf.random_normal_initializer(stddev=hyper.stddev))
        with tf.variable_scope("GAP"):
            gap_w = tf.get_variable("W", shape=(512, 257),
                    initializer=tf.random_normal_initializer(stddev=hyper.stddev))

        class_prob = tf.matmul(gap, gap_w)

        # print_model_params()
        return x, gap, class_prob

    def p(self,t):
        print (t.name, t.get_shape())

    def get_classmap(self, class_, conv_last):
        with tf.variable_scope("GAP", reuse=True):
            class_w = tf.gather(tf.transpose(tf.get_variable("W")), class_)
            class_w = tf.reshape(class_w, [-1, cnn_param.last_features, 1]) 
        conv_last_ = tf.image.resize_bilinear(conv_last, [hyper.image_h, hyper.image_w])
        conv_last_ = tf.reshape(conv_last_, [-1, hyper.image_h*hyper.image_w, cnn_param.last_features]) 
        classmap   = tf.reshape(tf.matmul(conv_last_, class_w), [-1, hyper.image_h,hyper.image_w])
        return classmap

    def _residual_block_first(self, x, out_channel, strides, name="unit"):
        in_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)

            # Shortcut connection
            if in_channel == out_channel:
                if strides == 1:
                    shortcut = tf.identity(x)
                else:
                    shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
            else:
                shortcut = self._conv(x, 1, out_channel, strides, name='shortcut')
            # Residual
            x = self._conv(x, 3, out_channel, strides, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, out_channel, 1, name='conv_2')
            x = self._bn(x, name='bn_2')
            # Merge
            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    def _residual_block(self, x, input_q=None, output_q=None, name="unit"):
        num_channel = x.get_shape().as_list()[-1]
        with tf.variable_scope(name) as scope:
            print('\tBuilding residual unit: %s' % scope.name)
            # Shortcut connection
            shortcut = x
            # Residual
            x = self._conv(x, 3, num_channel, 1, input_q=input_q, output_q=output_q, name='conv_1')
            x = self._bn(x, name='bn_1')
            x = self._relu(x, name='relu_1')
            x = self._conv(x, 3, num_channel, 1, input_q=output_q, output_q=output_q, name='conv_2')
            x = self._bn(x, name='bn_2')

            x = x + shortcut
            x = self._relu(x, name='relu_2')
        return x

    # Helper functions(counts FLOPs and number of weights)
    def _conv(self, x, filter_size, out_channel, stride, pad="SAME", input_q=None, output_q=None, name="conv"):
        b, h, w, in_channel = x.get_shape().as_list()
        x = utils._conv(x, filter_size, out_channel, stride, pad, input_q, output_q, name)
        f = 2 * (h/stride) * (w/stride) * in_channel * out_channel * filter_size * filter_size
        w = in_channel * out_channel * filter_size * filter_size
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _fc(self, x, out_dim, input_q=None, output_q=None, name="fc"):
        b, in_dim = x.get_shape().as_list()
        x = utils._fc(x, out_dim, input_q, output_q, name)
        f = 2 * (in_dim + 1) * out_dim
        w = (in_dim + 1) * out_dim
        scope_name = tf.get_variable_scope().name + "/" + name
        self._add_flops_weights(scope_name, f, w)
        return x

    def _bn(self, x, name="bn"):
        x = utils._bn(x, self.is_train, self._global_step, name)
        # f = 8 * self._get_data_size(x)
        # w = 4 * x.get_shape().as_list()[-1]
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, w)
        return x

    def _relu(self, x, name="relu"):
        x = utils._relu(x, 0.0, name)
        # f = self._get_data_size(x)
        # scope_name = tf.get_variable_scope().name + "/" + name
        # self._add_flops_weights(scope_name, f, 0)
        return x

    def _get_data_size(self, x):
        return np.prod(x.get_shape().as_list()[1:])

    def _add_flops_weights(self, scope_name, f, w):
        if scope_name not in self._counted_scope:
            self._flops += f
            self._weights += w
            self._counted_scope.append(scope_name)

if __name__ == '__main__':
    global_step = tf.Variable(0, trainable=False, name='global_step')
    model = CNN(global_step)
    images_tf = tf.placeholder(tf.float32, [None, hyper.image_h, hyper.image_w, hyper.image_c], name="images")
    _, _, prob_tf = model.build(images_tf)