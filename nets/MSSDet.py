import keras.backend as K
import numpy as np
from keras.layers import InputSpec, Layer
from keras.layers import (Activation, Concatenate, Conv2D, Flatten, Input,
                          Reshape, Add, AveragePooling2D,concatenate)
from keras.models import Model

from nets.detnet import detnet_59
from nets.subpixel import SubpixelConv2D


class Normalize(Layer):
    def __init__(self, scale, **kwargs):
        self.axis = 3
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))
        self.trainable_weights = [self.gamma]

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output


def MSSDet_model(input_shape=(224, 224, 3), num_classes=21):
    inputs = Input(input_shape, name='inputs')
    filters_list = [[64],
                    [64, 64, 256],
                    [128, 128, 512],
                    [256, 256, 1024],
                    [256, 256, 256],
                    [256, 256, 256]]
    blocks_list = [1, 3, 4, 6, 3, 3]

    #------------------------------------------------------------------------#
    #   net变量里面包含了整个detnet_59的结构，通过层名可以找到对应的特征层
    #------------------------------------------------------------------------#
    net = detnet_59(inputs, filters_list, blocks_list, num_classes)

    #-----------------------将提取到的主干特征进行处理---------------------------#
    num_anchors = 4
    # res1层
    net['res1_mbox_loc'] = Conv2D(
        num_anchors * 4, kernel_size=(3, 3), padding='same', name='res1_mbox_loc')(net['res1'])
    net['res1_mbox_loc_flat'] = Flatten(
        name='res1_mbox_loc_flat')(net['res1_mbox_loc'])
    net['res1_mbox_conf'] = Conv2D(num_anchors * num_classes, kernel_size=(
        3, 3), padding='same', name='res1_mbox_conf')(net['res1'])
    net['res1_mbox_conf_flat'] = Flatten(
        name='res1_mbox_conf_flat')(net['res1_mbox_conf'])

    # res1+res2+res3层
    num_anchors = 6
    net['res3_subpixel'] = SubpixelConv2D(upsampling_factor=2)(net['res3'])
    net['res3_subpixel'] = Conv2D(256, (1, 1))(net['res3_subpixel'])
    net['res1_passthrough'] = Reshape((56, 56, 256))(net['res1'])
    net['res2_opz'] = Add()([net['res1_passthrough'], net['res3_subpixel'], net['res2']])
    net['res2_opz_mbox_loc'] = Conv2D(num_anchors * 4, kernel_size=(
        3, 3), padding='same', name='res2_mbox_loc')(net['res2_opz'])
    net['res2_opz_mbox_loc_flat'] = Flatten(
        name='res2_opz_mbox_loc_flat')(net['res2_opz_mbox_loc'])
    net['res2_opz_mbox_conf'] = Conv2D(num_anchors * num_classes, kernel_size=(
        3, 3), padding='same', name='res2_opz_mbox_conf')(net['res2_opz'])
    net['res2_opz_mbox_conf_flat'] = Flatten(
        name='res2_opz_mbox_conf_flat')(net['res2_opz_mbox_conf'])

    # res2+res3+res4层
    num_anchors = 6
    net['res4_subpixel'] = SubpixelConv2D(upsampling_factor=2)(net['res4'])
    net['res4_subpixel'] = Conv2D(512, (1, 1))(net['res4_subpixel'])
    net['res2_passthrough'] = Reshape((28, 28, 1024))(net['res2'])
    net['res2_passthrough'] = Conv2D(512, (1, 1))(net['res2_passthrough'])
    net['res3_opz'] = Add()(
        [net['res2_passthrough'], net['res4_subpixel'], net['res3']])
    net['res3_opz_mbox_loc'] = Conv2D(num_anchors * 4, kernel_size=(
        3, 3), padding='same', name='res3_opz_mbox_loc')(net['res3_opz'])
    net['res3_opz_mbox_loc_flat'] = Flatten(
        name='res3_opz_mbox_loc_flat')(net['res3_opz_mbox_loc'])
    net['res3_opz_mbox_conf'] = Conv2D(num_anchors * num_classes, kernel_size=(
        3, 3), padding='same', name='res3_opz_mbox_conf')(net['res3_opz'])
    net['res3_opz_mbox_conf_flat'] = Flatten(
        name='res3_opz_mbox_conf_flat')(net['res3_opz_mbox_conf'])

    # res3+res4+dires5
    num_anchors = 6
    net['dires5_subpixel'] = SubpixelConv2D(upsampling_factor=2)(net['dires5'])
    net['dires5_subpixel'] = Conv2D(1024, (1, 1),strides=2)(net['dires5_subpixel'])
    net['res3_passthrough'] = Reshape((14, 14, 2048))(net['res3'])
    net['res3_passthrough'] = Conv2D(1024, (1, 1))(net['res3_passthrough'])
    net['res4_opz'] = Add()(
        [net['res3_passthrough'], net['dires5_subpixel'], net['res4']])
    net['res4_opz_mbox_loc'] = Conv2D(num_anchors * 4, kernel_size=(
        3, 3), padding='same', name='res4_opz_mbox_loc')(net['res4_opz'])
    net['res4_opz_mbox_loc_flat'] = Flatten(
        name='res4_opz_mbox_loc_flat')(net['res4_opz_mbox_loc'])
    net['res4_opz_mbox_conf'] = Conv2D(num_anchors * num_classes, kernel_size=(
        3, 3), padding='same', name='res4_opz_mbox_conf')(net['res4_opz'])
    net['res4_opz_mbox_conf_flat'] = Flatten(
        name='res4_opz_mbox_conf_flat')(net['res4_opz_mbox_conf'])

    # res4+dires5+dires6
    num_anchors = 4
    net['dires6_subpixel'] = SubpixelConv2D(upsampling_factor=2)(net['dires6'])
    net['dires6_subpixel'] = Conv2D(256, (1, 1),strides=2)(net['dires6_subpixel'])
    net['res4_passthrough'] = Conv2D(256, (1, 1))(net['res4'])
    net['dires5_opz'] = Add()(
        [net['res4_passthrough'], net['dires6_subpixel'], net['dires5']])
    net['dires5_opz_mbox_loc'] = Conv2D(num_anchors * 4, kernel_size=(
        3, 3), padding='same', name='dires5_opz_mbox_loc')(net['dires5_opz'])
    net['dires5_opz_mbox_loc_flat'] = Flatten(
        name='dires5_opz_mbox_loc_flat')(net['dires5_opz_mbox_loc'])
    net['dires5_opz_mbox_conf'] = Conv2D(num_anchors * num_classes, kernel_size=(
        3, 3), padding='same', name='dires5_opz_mbox_conf')(net['dires5_opz'])
    net['dires5_opz_mbox_conf_flat'] = Flatten(
        name='dires5_opz_mbox_conf_flat')(net['dires5_opz_mbox_conf'])

    # dires6
    num_anchors = 4
    net['dires6_mbox_loc'] = Conv2D(
        num_anchors * 4, kernel_size=(3, 3), padding='same', name='dires6_mbox_loc')(net['dires6'])
    net['dires6_mbox_loc_flat'] = Flatten(
        name='dires6_mbox_loc_flat')(net['dires6_mbox_loc'])
    net['dires6_mbox_conf'] = Conv2D(num_anchors * num_classes, kernel_size=(
        3, 3), padding='same', name='dires6_mbox_conf')(net['dires6'])
    net['dires6_mbox_conf_flat'] = Flatten(
        name='dires6_mbox_conf_flat')(net['dires6_mbox_conf'])


    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc')([net['res1_mbox_loc_flat'],
                                                            net['res2_opz_mbox_loc_flat'],
                                                            net['res3_opz_mbox_loc_flat'],
                                                            net['res4_opz_mbox_loc_flat'],
                                                            net['dires5_opz_mbox_loc_flat'],
                                                            net['dires6_mbox_loc_flat']])

    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf')([net['res1_mbox_conf_flat'],
                                                              net['res2_opz_mbox_conf_flat'],
                                                              net['res3_opz_mbox_conf_flat'],
                                                              net['res4_opz_mbox_conf_flat'],
                                                              net['dires5_opz_mbox_conf_flat'],
                                                              net['dires6_mbox_conf_flat']])
    # 8732,4
    net['mbox_loc'] = Reshape((-1, 4), name='mbox_loc_final')(net['mbox_loc'])
    # 8732,21
    net['mbox_conf'] = Reshape(
        (-1, num_classes), name='mbox_conf_logits')(net['mbox_conf'])
    net['mbox_conf'] = Activation(
        'softmax', name='mbox_conf_final')(net['mbox_conf'])
    # 8732,25
    net['predictions'] = Concatenate(
        axis=-1, name='predictions')([net['mbox_loc'], net['mbox_conf']])

    model = Model(inputs=inputs, outputs=net['predictions'])
    return model
