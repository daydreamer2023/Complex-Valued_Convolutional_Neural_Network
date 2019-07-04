# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
#import numpy as np
from np import *
from functions import *
from collections import OrderedDict
from layers import *
import numpy
from util import *

pool = True


class Network:
    def __init__(self, input_dim=(2, 28, 28), conv_stride=1, conv_pad=0, pool_size=2, pool_stride=2,
                 conv_param_1 = {'filter_num':9, 'filter_size':23, 'pool_or_not':True, 'pool':'max'},
                 conv_param_2 = {'filter_num':9, 'filter_size':27, 'pool_or_not':True, 'pool':'max'},
                 conv_param_3 = {'filter_num':16, 'filter_size':9, 'pool_or_not':True, 'pool':'max'},
                 hidden_size_1=16, hidden_size_2=32, hidden_size_3=64, output_size=10, class_size=10,
                 c_weight_init_mean=0.1, c_weight_init_std=0.01, weight_init_mean=0.1, weight_init_std=0.01,
                 learning_rate={"lr_a":0.01, "lr_p":0.01}, c_learning_rate={"lr_a":0.01, "lr_p":0.01},
                 set_lr = False, optimizer='SGD', he=True, mag=1.0, pool='max', bias=False, input_layer=False, memory_save=False):
        # 重みの初期化===========
        self.prepr = input_layer
        conv_list = [conv_param_2]
        hidden_list = []
        self.bias = bias
        self.input_dim = input_dim
        self.params = {}
        self.conv_num = 0
        self.hidden_num = 0
        pre_channel_num = input_dim[0]
        output_layer_idx = 1    # 最終層は何番目の層か
        conv_out_h, conv_out_w = input_dim[1], input_dim[2]


        self.he = he

        for idx, conv_param in enumerate(conv_list):
            if self.he:
                self.params['W' + str(idx+1)] = (np.abs(np.sqrt(2.0*np.pi / pre_channel_num)/conv_param['filter_size']*np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])) \
                                                * np.exp(1.j * 2*np.pi * np.random.rand(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size']))).astype('complex64')
            else:
                self.params['W' + str(idx+1)] = (np.abs((c_weight_init_std*np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])) + c_weight_init_mean) \
                                                * np.exp(1.j * 2*np.pi * np.random.rand(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size']))).astype('complex64')
            if bias:
                self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'], dtype='complex64')
            else:
                self.params['b' + str(idx+1)] = None

            pre_channel_num = conv_param['filter_num']
            conv_out_h = conv_output_size(conv_out_h, conv_param['filter_size'], conv_stride, conv_pad)
            conv_out_w = conv_output_size(conv_out_w, conv_param['filter_size'], conv_stride, conv_pad)
            if conv_param['pool_or_not']:
                conv_out_h = conv_output_size(conv_out_h, pool_size, pool_stride, 0)
                conv_out_w = conv_output_size(conv_out_w, pool_size, pool_stride, 0)
            output_layer_idx += 1
            self.conv_num += 1

        conv_out_size = int(pre_channel_num * conv_out_h * conv_out_w)
        hid_idx = output_layer_idx

        for idx, hid_size in enumerate(hidden_list):
            if self.he:
                self.params['W'+str(hid_idx + idx)] = (np.abs((np.sqrt(2.0*np.pi/conv_out_size) * np.random.randn( hid_size, conv_out_size ))) * np.exp(1.j * 2*np.pi * np.random.rand(hid_size, conv_out_size))).astype('complex64')
            else:
                self.params['W'+str(hid_idx + idx)] = (np.abs((weight_init_std * np.random.randn( hid_size, conv_out_size ) + weight_init_mean)) * np.exp(1.j * 2*np.pi * np.random.rand(hid_size, conv_out_size))).astype('complex64')
            if bias:
                self.params['b'+str(hid_idx + idx)] = np.zeros(hid_size, dtype='complex64')
            else:
                self.params['b'+str(hid_idx + idx)] = None
            output_layer_idx += 1
            self.hidden_num += 1
            conv_out_size = hid_size


        if self.he:
            self.params['W'+str(output_layer_idx)] = (np.abs((np.sqrt(2.0*np.pi/conv_out_size) * np.random.randn( output_size, conv_out_size ))) * np.exp(1.j * 2*np.pi * np.random.rand(output_size, conv_out_size))).astype('complex64')
        else:
            self.params['W'+str(output_layer_idx)] = (np.abs((weight_init_std * np.random.randn( output_size, conv_out_size ) + weight_init_mean)) * np.exp(1.j * 2*np.pi * np.random.rand(output_size, conv_out_size))).astype('complex64')
        if bias:
            self.params['b'+str(output_layer_idx)] = np.zeros(output_size, dtype='complex64')
        else:
            self.params['b'+str(output_layer_idx)] = None
        #self.params['W'+str(output_layer_idx+1)] = weight_init_std * np.abs( np.random.randn( output_size, hidden_size ) ) * np.exp(1.j * 2*np.pi * np.random.rand(output_size, hidden_size))

        # レイヤの生成===========
        if self.prepr:
            self.input_layer = InputLayer()
        lr_a, lr_p = learning_rate["lr_a"], learning_rate["lr_p"]
        c_lr_a, c_lr_p = c_learning_rate["lr_a"], c_learning_rate["lr_p"]
        self.layers = []
        for i , conv_param in enumerate(conv_list):
            self.layers.append(ConvPool(W=self.params['W'+str(i+1)], conv_stride=conv_stride, conv_pad=conv_pad,
                                        bias=bias, b=self.params['b'+str(i+1)], lr_a=c_lr_a, lr_p=c_lr_p, pool_h=pool_size, pool_w=pool_size, pool_stride=pool_stride,
                                        optimizer=optimizer, batchnorm=False, mag=mag, pool_or_not=conv_param['pool_or_not'], pool=conv_param['pool'], memory_save=memory_save))
        for i in range(self.hidden_num):
            self.layers.append(AffineTanh(self.params['W'+str(hid_idx + i)], bias=bias, b=self.params['b'+str(hid_idx + i)], lr_a=lr_a, lr_p=lr_p, optimizer=optimizer, batchnorm=False, mag=mag))

        #self.layers.append(ConvPool(W=self.params['W2'], conv_stride=conv_param_2['stride'], conv_pad=conv_param_2['pad'], lr_a=c_lr_a, lr_p=c_lr_p, pool_h=pool_size, pool_w=pool_size, pool_stride=pool_stride))
        #self.layers.append(AffineTanh(self.params['W'+str(output_layer_idx)], lr_a=lr_a, lr_p=lr_p))
        self.last_layer = AffineTanh(self.params['W'+str(output_layer_idx)], bias=bias, b=self.params['b'+str(output_layer_idx)], lr_a=lr_a, lr_p=lr_p, optimizer=optimizer, batchnorm=False, mag=mag)

        self.out_size = int(output_size)
        self.class_size = int(class_size)

    def predict(self, x):
        if self.prepr:
            x = self.input_layer.forward(x)
        for layer in self.layers:
            x = layer.forward(x)
        x = self.last_layer.forward(x)
        return x

    def teacher_label(self): # 教師ラベルt0,t1,...,tn T=(t0 t1 ... tn)を出力
        t = np.arange(self.class_size)
        T = to_one_hot_label(t, self.out_size, self.class_size)    # 教師ラベルt0,t1,...,tn T=(t0 t1 ... tn)
        return T

    def classify(self, x, T):  # predict()の出力がどのクラスか（どの教師ラベルに近いか）を調べる
        E = numpy.zeros((x.shape[0], T.shape[1]))   # E[i] = loss(predict(x), ti)

        y = to_cpu(self.predict(x))
        for i in range(T.shape[1]):
            E[:,i] = mean_squared_error_np_b(y, T[:,i])

        clss = numpy.argmin(E, axis=1)
        return clss

    def set_lr(self, delta_loss):
        for layer in self.layers:
            layer.set_lr(delta_loss)
        self.last_layer.set_lr(delta_loss)

    def get_lr(self):
        lr = {}
        lr_a = self.layers[0].lr_a
        lr_p = self.layers[0].lr_p
        lr['lr_a'] = lr_a
        lr['lr_p'] = lr_p
        return lr

    def loss(self, x, t):
        y = self.predict(x)
        y = mean_squared_error(y, t)
        return y

    def accuracy(self, x, t, batch_size=100):
        acc = 0.0
        T = to_cpu(self.teacher_label())
        #print(x.shape)
        #print(t.shape)
        channel, row, col = self.input_dim
        y = self.classify(x, T)
        #tt = np.argmax(t,axis=0)
        #print(y)
        #print(tt)
        acc = y[y==t].size
        #print(y)
        #print(t)

        return acc / x.shape[0]

    def gradient(self, x, t, return_loss=False):
        # forward
        loss = self.loss(x, t)

        # backward
        #t_vec = to_one_hot_label(t, self.out_size)
        t_vec = t
        t_0 = self.last_layer.backward(t_vec)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            t_0 = layer.backward(t_0)

        # 設定
        '''
        grads_a = {}
        grads_p = {}
        '''
        '''
        for i, layer_idx in enumerate((0, 1)):
            grads_a['W' + str(i+1)] = self.layers[layer_idx].dWa
            grads_p['W' + str(i+1)] = self.layers[layer_idx].dWp

        grads_a['W3'] = self.last_layer.dWa
        grads_p['W3'] = self.last_layer.dWp
        '''
        if return_loss:
            return loss

    '''
    def save_params(self, file_name="params.pkl"):
        params = []
        layernum = len(self.layers)
        for layer in range(layernum):
            params.append( self.layers[layer].return_W() )
        params.append(self.last_layer.return_W())
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        layernum = len(self.layers)
        for layer in range(layernum):
            self.layers[layer].set_W(params[layer])
        self.last_layer.set_W(params[layernum])

        #layernum = len(self.layers)
        #for i, layer_idx in enumerate(np.arange(layernum)):
        #    self.layers[layer_idx].W = self.params['W' + str(i+1)]

    '''
    def save_params(self, file_name="params.pkl"):
        params_W = []
        params_b = []

        layernum = len(self.layers)
        for layer in range(layernum):
            params_W.append( self.layers[layer].return_W() )
            if self.bias:
                params_b.append( self.layers[layer].return_b() )

        params_W.append(self.last_layer.return_W())
        if self.bias:
            params_b.append( self.last_layer.return_b() )

        params = {}
        params['W'] = params_W
        params['b'] = params_b
        params['conv_num'] = self.conv_num
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        params_W = params['W']
        params_b = params['b']
        layernum = len(self.layers)
        for layer in range(layernum):
            self.layers[layer].set_W(params_W[layer])
            if self.bias:
                self.layers[layer].set_b(params_b[layer])
        self.last_layer.set_W(params_W[layernum])
        if self.bias:
            self.last_layer.set_b(params_b[layernum])
