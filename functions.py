# coding: utf-8
#import numpy as np
from np import *
import numpy
from util import *

def activation(x):
    return np.tanh(np.abs(x)) * np.exp(1.j * np.angle(x))

def softmax(z):
    if z.ndim == 2:
        #z = z.T
        z = z - np.max(z, axis=0)
        y = np.exp(np.abs(z)) / np.sum(np.exp(np.abs(z)), axis=0)
        return y

    z = z - np.max(np.abs(z)) # オーバーフロー対策
    return np.exp(np.abs(z)) / np.sum(np.exp(np.abs(z)))

'''
def to_one_hot_label(X, outsize):
    T = np.zeros((X.size,outsize),dtype='int')
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    T = T.T
    return T
'''

def to_one_hot_label(X, outsize, classsize):
    if outsize < classsize:
        T = to_one_hot_label2(X, outsize)
    else:
        T = to_one_hot_label1(X, outsize)
    return T

def to_one_hot_label1(X, outsize):
    T = -np.ones((X.size, outsize),dtype='complex128')
    for i in range(X.size):
        T[i, X[i]] = 1
    T = T.T
    return T

def to_one_hot_label2(X, outsize):
    T = -np.ones((X.size, outsize),dtype='complex128')
    for i in range(X.size):
        if X[i] == 0:
            T[i] = 0.01
        else:
            T[i, X[i]-1] = 1
    T = T.T
    return T

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    #print(t.shape)
    #print(y.shape)
    if t.size == y.size:
        t = t.argmax(axis=0)

    batch_size = y.shape[1]
    return -np.sum(np.log(y[t,np.arange(batch_size)])) / batch_size

def mean_squared_error(y, t):
    batch_size = y.shape[1]
    output_size = y.shape[0]
    return np.sum( np.abs(y - t) **2 ) / output_size / batch_size

def mean_squared_error_np(y, t):
    y = to_cpu(y)
    t = to_cpu(t)
    batch_size = y.shape[1]
    output_size = y.shape[0]
    return numpy.sum( numpy.abs(y - t) **2 ) / output_size / batch_size

def mean_squared_error_np_b(y, t):
    y = to_cpu(y)
    t = to_cpu(t)
    output_size = y.shape[0]
    return numpy.sum( numpy.abs(y.T - t) **2 , axis=1) / output_size

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w), dtype='complex128')

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 入力データの形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1), dtype='complex128')
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1
