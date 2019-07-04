# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
#import numpy as np
#import numpy.matlib
from np import *
from functions import *
from util import im2col, col2im, g_repmat

backpro = 0
cbp = 0
beta1=0.9
beta2=0.999

class AffineTanh:
    def __init__(self, W, lr_a, lr_p, bias=False, b=None, mag=1.0, optimizer='SGD', batchnorm=False):
        self.W =W.astype('complex64')
        self.bias = bias
        if self.bias:
            self.b = b
        else:
            self.b = 0

        self.x = None
        self.original_x_shape = None
        self.u = None
        self.y = None    #x->(Affine)->u->(Tanh)->y
        self.mag = mag
        # 重みの微分
        self.dWa = None
        self.dWp = None
        if self.bias:
            self.dba = None
            self.dbp = None

        self.lr_a = lr_a
        self.lr_p = lr_p

        self.optimizer = optimizer
        self.iter = 0
        self.m = None
        self.v = None
        if self.bias:
            self.bm = None
            self.bv = None

        self.batchnorm = batchnorm
        self.batch_size = None
        self.out_size = None
        self.u_shape = None
        self.mu = None
        self.min = None
        self.sigma_sq = None
        self.epsilon = 1e-7
        self.v = None
        self.v_amp = None

        self.count = 0

    def set_lr(self, delta_loss):
        self.lr_a = self.lr_a * (1/(1+delta_loss))
        self.lr_p = self.lr_p * (1/(1+delta_loss))

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        if len(self.original_x_shape) == 4:
            x = x.T
        self.x = x
        u = (np.dot(self.W, self.x).T + self.b).T
        self.u = u
        y = activation(self.mag*u)
        self.y = y

        if self.batchnorm:
            self.out_size, self.batch_size = self.y.shape
            self.mu = np.abs(self.y).mean(axis=1).reshape((self.out_size, -1))
            self.mu = g_repmat(self.mu, 1, self.batch_size)
            self.min = np.abs(self.y).min(axis=1).reshape((self.out_size, -1))
            self.min = g_repmat(self.min, 1, self.batch_size)
            self.sigma_sq = np.mean((np.abs(self.y)-self.mu)**2, axis=1).reshape((self.out_size, -1))
            self.sigma_sq = g_repmat(self.sigma_sq, 1, self.batch_size)
            self.v_amp = (np.abs(self.y) - self.min + self.epsilon) / np.sqrt(self.sigma_sq + self.epsilon)
            y = self.v_amp*np.exp(1.j * np.angle(self.y))
            '''
            if self.count == 0:
                print(self.y)
                print(self.mu)
                print(self.min)
                print(self.sigma_sq)
                print(self.v_amp)
                print(y)
                self.count += 1
            '''


        #print(self.x.shape)
        #print(self.y.shape)
        return y

    def backward(self, t): # この層の教師出力tから前層の教師出力t_0を計算
        if self.batchnorm:
            t = (np.sqrt(self.sigma_sq + self.epsilon) * np.abs(t)  + self.min - self.epsilon) * np.exp(1.j * np.angle(t))

        if backpro == 0:
            t_tmp = np.dot( np.conj(t.T) , self.W)
            t_0 = activation( self.mag*np.conj( t_tmp.T ) )
            # t_0をxの形状に戻す処理
            if len(self.original_x_shape) == 4:
                t_0 = t_0.T
                t_0 = t_0.reshape(*self.original_x_shape)


        '''
        lis = []
        for i in range(self.x.shape[1]):
            lis.append([i]*self.W.shape[1])
        lis = [flatten for inner in lis for flatten in inner]
        y_b = g_repmat(self.y, 1, self.W.shape[1])[:,lis].reshape((self.x.shape[1], self.W.shape[0], self.W.shape[1]))
        t_b = g_repmat(t, 1, self.W.shape[1])[:,lis].reshape((self.x.shape[1], self.W.shape[0], self.W.shape[1]))
        u_b = g_repmat(self.u, 1, self.W.shape[1])[:,lis].reshape((self.x.shape[1], self.W.shape[0], self.W.shape[1]))

        lis = []
        for i in range(self.x.shape[1]):
            lis.append([i]*self.W.shape[0])
        lis = [flatten for inner in lis for flatten in inner]
        x_b = g_repmat(self.x.T, self.W.shape[0], 1)[lis].reshape((self.x.shape[1], self.W.shape[0], self.W.shape[1]))

        theta_b = np.angle(y_b) - np.angle(x_b) - np.angle(self.W)

        self.dWa = self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.cos(theta_b) \
                            - np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / np.abs(u_b) * np.sin(theta_b))
        self.dWp = self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.sin(theta_b) \
                            + np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / np.abs(u_b) * np.cos(theta_b))
        self.dWa = np.sum(self.dWa, axis=0)
        self.dWp = np.sum(self.dWp, axis=0)
        '''


        self.dWa = np.zeros(self.W.shape, dtype='float32')
        self.dWp = np.zeros(self.W.shape, dtype='float32')

        for b in range(self.x.shape[1]):
            y_b = g_repmat(self.y[:,b].reshape(self.y[:,b].shape[0],1), 1, self.W.shape[1])
            t_b = g_repmat(t[:,b].reshape(t[:,b].shape[0],1), 1, self.W.shape[1])
            u_b = g_repmat(self.u[:,b].reshape(self.u[:,b].shape[0],1), 1, self.W.shape[1])
            x_b = g_repmat(self.x[:,b].T, self.W.shape[0], 1)
            theta_b = np.angle(y_b) - np.angle(x_b) - np.angle(self.W)
            self.dWa += self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.cos(theta_b) \
                                - np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-10) * np.sin(theta_b))
            self.dWp += self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.sin(theta_b) \
                                + np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-10) * np.cos(theta_b))

        #self.dWa /= np.sqrt(self.y.shape[1])
        #self.dWp /= np.sqrt(self.y.shape[1])
        if self.bias:
            y_b = self.y
            t_b = t
            u_b = self.u
            x_b = np.ones((self.b.shape[0], y_b.shape[1]), dtype='complex64')
            b_b = self.b.reshape((self.b.shape[0], 1))
            theta_b = np.angle(y_b) - np.angle(b_b)
            self.dba = self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.cos(theta_b) \
                                - np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-10) * np.sin(theta_b))
            self.dbp = self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.sin(theta_b) \
                                + np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-10) * np.cos(theta_b))
            self.dba = np.sum(self.dba, axis=1)# / np.sqrt(self.y.shape[1])
            self.dbp = np.sum(self.dbp, axis=1)# / np.sqrt(self.y.shape[1])

        '''
        if self.bias:
            self.dba = np.zeros((self.b.shape[0], 1), dtype='float32')
            self.dbp = np.zeros((self.b.shape[0], 1), dtype='float32')
            for b in range(self.x.shape[1]):
                y_b = self.y[:,b].reshape(self.y[:,b].shape[0],1)
                t_b = t[:,b].reshape(t[:,b].shape[0],1)
                u_b = self.u[:,b].reshape(self.u[:,b].shape[0],1)
                x_b = g_repmat(1.+0.j, self.b.shape[0], 1)
                b_b = self.b.reshape((self.b.shape[0], 1))
                theta_b = np.angle(y_b) - np.angle(x_b) - np.angle(b_b)
                self.dba += self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.cos(theta_b) \
                                    - np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / np.abs(u_b) * np.sin(theta_b))
                self.dbp += self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.sin(theta_b) \
                                    + np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / np.abs(u_b) * np.cos(theta_b))
            self.dba = self.dba.flatten()
            self.dbp = self.dbp.flatten()
        '''
        '''
        # 多重for文は計算がとても遅い
        for n in range(self.x.shape[1]):
            for j in range(self.W.shape[0]):
                for i in range(self.W.shape[1]):
                    theta = np.angle(self.y[j,n]) - np.angle(self.x[i,n]) - np.angle(self.W[j,i])
                    self.dWa[j,i] += (1 - np.abs(self.y[j,n])**2) * (np.abs(self.y[j,n]) - np.abs(t[j,n])*np.cos(np.angle(self.y[j,n] - np.angle(t[j,n])))) * np.abs(self.x[i,n]) * np.cos(theta) \
                                        - np.abs(self.y[j,n]) * np.abs(t[j,n]) * np.sin(np.angle(self.y[j,n] - np.angle(t[j,n]))) * np.abs(self.x[i,n]) / np.abs(self.u[j,n]) * np.sin(theta)
                    self.dWp[j,i] += (1 - np.abs(self.y[j,n])**2) * (np.abs(self.y[j,n]) - np.abs(t[j,n])*np.cos(np.angle(self.y[j,n] - np.angle(t[j,n])))) * np.abs(self.x[i,n]) * np.sin(theta) \
                                        + np.abs(self.y[j,n]) * np.abs(t[j,n]) * np.sin(np.angle(self.y[j,n] - np.angle(t[j,n]))) * np.abs(self.x[i,n]) / np.abs(self.u[j,n]) * np.cos(theta)
        '''
        if self.optimizer == 'Adam':
            if self.m is None:
                self.m, self.v = {}, {}
                self.m["dWa"] = np.zeros_like(self.dWa, dtype='float32')
                self.m["dWp"] = np.zeros_like(self.dWp, dtype='float32')
                self.v["dWa"] = np.zeros_like(self.dWa, dtype='float32')
                self.v["dWp"] = np.zeros_like(self.dWp, dtype='float32')
            self.iter += 1
            lr_t_a  = self.lr_a * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
            lr_t_p  = self.lr_p * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
            self.m["dWa"] += (1 - beta1) * (self.dWa - self.m["dWa"])
            self.m["dWp"] += (1 - beta1) * (self.dWp - self.m["dWp"])
            self.v["dWa"] += (1 - beta2) * (self.dWa**2 - self.v["dWa"])
            self.v["dWp"] += (1 - beta2) * (self.dWp**2 - self.v["dWp"])

            newWabs = np.abs(self.W) - lr_t_a * self.m["dWa"] / (np.sqrt(self.v["dWa"]) + 1e-7)
            newWphase = np.angle(self.W) - lr_t_p * self.m["dWp"] / (np.sqrt(self.v["dWp"]) + 1e-7)

            if self.bias:
                if self.bm is None:
                    self.bm, self.bv = {}, {}
                    self.bm["dba"] = np.zeros_like(self.dba, dtype='float32')
                    self.bm["dbp"] = np.zeros_like(self.dbp, dtype='float32')
                    self.bv["dba"] = np.zeros_like(self.dba, dtype='float32')
                    self.bv["dbp"] = np.zeros_like(self.dbp, dtype='float32')
                lr_t_a  = self.lr_a * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
                lr_t_p  = self.lr_p * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
                self.bm["dba"] += (1 - beta1) * (self.dba - self.bm["dba"])
                self.bm["dbp"] += (1 - beta1) * (self.dbp - self.bm["dbp"])
                self.bv["dba"] += (1 - beta2) * (self.dba**2 - self.bv["dba"])
                self.bv["dbp"] += (1 - beta2) * (self.dbp**2 - self.bv["dbp"])

                newbabs = np.abs(self.b) - (lr_t_a * self.bm["dba"] / (np.sqrt(self.bv["dba"]) + 1e-7)).flatten()
                newbphase = np.angle(self.b) - (lr_t_p * self.bm["dbp"] / (np.sqrt(self.bv["dbp"]) + 1e-7)).flatten()

        elif self.optimizer == 'SGD':
            newWabs = np.abs(self.W) - self.lr_a * self.dWa
            newWphase = np.angle(self.W) - self.lr_p * self.dWp
            if self.bias:
                newbabs = np.abs(self.b) - self.lr_a * self.dba
                newbphase = np.angle(self.b) - self.lr_p * self.dbp
        #newWphase[newWabs < 0] += np.pi
        #newWabs[newWabs < 0] = -newWabs[newWabs < 0]
        self.W = newWabs * np.exp(1.j * newWphase)
        if self.bias:
            self.b = newbabs * np.exp(1.j * newbphase)
        #print(self.W[0,0])
        #print(str(np.abs(self.W[0,0])) + ' ' + str(np.angle(self.W[0,0])))
        if backpro == 1:
            t_tmp = np.dot( np.conj(t.T) , self.W)
            t_0 = activation( np.conj( t_tmp.T ) )
            # t_0をxの形状に戻す処理
            if len(self.original_x_shape) == 4:
                t_0 = t_0.T
                t_0 = t_0.reshape(*self.original_x_shape)

        return t_0

    def return_W(self):
        return self.W

    def set_W(self, W):
        self.W = W

    def return_b(self):
        return self.b

    def set_b(self,b):
        self.b = b

class AffineSoftmaxWithLoss:
    def __init__(self, W, lr_a=0.01, lr_p=0.01):
        self.W = W
        self.loss = None
        self.x = None
        self.original_x_shape = None
        self.u = None
        self.y = None    #x->(Affine)->u->(Tanh)->y
        self.t = None
        # 重みの微分
        self.dWa = None
        self.dWp = None

        self.lr_a = lr_a
        self.lr_p = lr_p

    def forward(self, x, t, entropy=True):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        if len(self.original_x_shape) == 4:
            x = x.T
        self.x = x
        #print(self.x.shape)
        #print(self.W.shape)
        u = np.dot(self.W, self.x)
        self.u = u
        self.y = softmax(u)

        if entropy:
            self.t = t
            self.loss = cross_entropy_error(self.y, self.t)
            return self.loss
        else:
            return self.y


    def backward(self, t):
        t_tmp = np.dot( np.conj(t.T) , self.W)
        t_0 = softmax(np.abs(t_tmp))
        t_0 = np.conj( t_0.T )
        # t_0をxの形状に戻す処理
        if len(self.original_x_shape) == 4:
            t_0 = t_0.T
            t_0 = t_0.reshape(*self.original_x_shape)

        #print(self.W.shape)
        #print(self.x.shape)
        #print(self.u.shape)
        #print(self.y.shape)
        #print(t.shape)
        self.dWa = np.zeros(self.W.shape, dtype='float32')
        self.dWp = np.zeros(self.W.shape, dtype='float32')

        for b in range(self.x.shape[1]):
            y_b = g_repmat(self.y[:,b].reshape(self.y[:,b].shape[0], 1), 1, self.W.shape[1])
            t_b = g_repmat(t[:,b].reshape(t[:,b].shape[0],1), 1, self.W.shape[1])
            u_b = g_repmat(self.u[:,b].reshape(self.u[:,b].shape[0],1), 1, self.W.shape[1])
            x_b = g_repmat(self.x[:,b].T, self.W.shape[0], 1)
            #print(self.W.shape)
            #print(x_b.shape)
            #print(u_b.shape)
            #print(y_b.shape)
            #print(t_b.shape)
            A = (y_b - t_b) * np.abs(x_b)
            theta_b = np.angle(u_b) - np.angle(x_b) - np.angle(self.W)
            self.dWa += A * np.cos(theta_b)
            self.dWp += A * np.sin(theta_b)
        newWabs = np.abs(self.W) - self.lr_a * self.dWa
        newWphase = np.angle(self.W) - self.lr_p * self.dWp
        #newWphase[newWabs < 0] += np.pi
        #newWabs[newWabs < 0] = -newWabs[newWabs < 0]
        self.W = newWabs * np.exp(1.j * newWphase)
        '''
        for n in range(self.y.shape[1]):
            for j in range(self.W.shape[0]):
                for i in range(self.W.shape[1]):
                    A = (self.y[j,n] - self.t[j,n]) * np.abs(self.x[i,n])
                    theta = np.angle(self.u[j,n]) - np.angle(self.x[i,n]) - np.angle(self.W[j,i])
                    self.dWa[j,i] += A * np.cos(theta)
                    self.dWp[j,i] += A * np.sin(theta)
        '''
        return t_0


class ConvPool:
    def __init__(self, W, bias=False, b=None, mag=1.0, pool_or_not=True, pool='max', pool_h=2, pool_w=2,
                    conv_stride=1, conv_pad=0, pool_stride=2, pool_pad=0, lr_a=0.01, lr_p=0.01,
                    optimizer='SGD', batchnorm=False, memory_save=False):
        self.W = W.astype('complex64')
        self.bias = bias
        if self.bias:
            self.b = b
        else:
            self.b = 0
        self.W_col = None   # im2colしたW
        self.conv_stride = conv_stride
        self.conv_pad = conv_pad
        self.pool_or_not = pool_or_not
        self.pool = pool
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.pool_stride = pool_stride
        self.pool_pad = pool_pad
        self.arg_max = None # Poolingで入力のどこがmaxだったかを保持

        self.mag=mag

        self.x = None       # 入力
        self.x_col = None   # im2colしたx
        self.u = None       # Convolutionを取った後
        self.u_col = None   # im2colしたu
        self.conv_y = None  # self.uを活性化関数に通した後
        self.conv_y_col = None # im2colしたself.conv_y
        self.pool_x_col = None # Poolingの入力(=conv_y)を（Pooling用の形状に）im2colしたもの
        self.pool_y = None  # self.conv_yをPoolingした後

        self.dWa = None
        self.dWp = None
        if self.bias:
            self.dba = None
            self.dbp = None

        self.lr_a = lr_a
        self.lr_p = lr_p

        self.optimizer = optimizer
        self.iter = 0
        self.m = None
        self.v = None
        if self.bias:
            self.bm = None
            self.bv = None
        self.batchnorm = batchnorm
        self.batch_size = None
        self.out_size = None
        self.u_shape = None
        self.mu = None
        self.min = None
        self.sigma_sq = None
        self.epsilon = 1e-7
        self.v = None
        self.v_amp = None

        self.memory_save = memory_save

    def set_lr(self, delta_loss):
        self.lr_a = self.lr_a * (1/(1+delta_loss))
        self.lr_p = self.lr_p * (1/(1+delta_loss))

    def forward(self, x):
        #print('a')
        # Convolution
        filternum, channel, filter_h, filter_w = self.W.shape
        batchsize, channel, height, width = x.shape
        conv_out_h = 1 + int((height + 2*self.conv_pad - filter_h) / self.conv_stride)
        conv_out_w = 1 + int((width + 2*self.conv_pad - filter_w) / self.conv_stride)

        self.x = x
        self.x_col = im2col(x, filter_h, filter_w, self.conv_stride, self.conv_pad)
        self.W_col = self.W.reshape(filternum, -1).T

        self.u_col = np.dot(self.x_col, self.W_col) + self.b
        self.u = self.u_col.reshape(batchsize, conv_out_h, conv_out_w, -1).transpose(0, 3, 1, 2)

        self.conv_y_col = activation(self.mag*self.u_col)
        out1 = self.conv_y_col

        if self.batchnorm:
            if self.batchnorm:
                self.batch_size, self.out_size  = self.conv_y_col.shape
                self.mu = np.abs(self.conv_y_col).mean(axis=0).reshape((1,-1))
                self.mu = g_repmat(self.mu, self.batch_size, 1)
                self.min = np.abs(self.conv_y_col).min(axis=0).reshape((1, -1))
                self.min = g_repmat(self.min, self.batch_size, 1)
                self.sigma_sq = np.mean((np.abs(self.conv_y_col)-self.mu)**2, axis=0).reshape((1, -1))
                self.sigma_sq = g_repmat(self.sigma_sq, self.batch_size, 1)
                self.v_amp = (np.abs(self.conv_y_col) - self.min + self.epsilon) / np.sqrt(self.sigma_sq + self.epsilon)
                out1 = self.v_amp*np.exp(1.j * np.angle(self.conv_y_col))

        self.conv_y = out1.reshape(batchsize, conv_out_h, conv_out_w, -1).transpose(0, 3, 1, 2)
        out = self.conv_y
        # Pooling
        if self.pool_or_not:
            batchsize, channel, height, width = self.conv_y.shape
            pool_out_h = 1 + int((height +2*self.pool_pad - self.pool_h) / self.pool_stride)
            pool_out_w = 1 + int((width +2*self.pool_pad - self.pool_w) / self.pool_stride)

            self.pool_x_col = im2col(self.conv_y, self.pool_h, self.pool_w, self.pool_stride, self.pool_pad)
            self.pool_x_col = self.pool_x_col.reshape(-1, self.pool_h * self.pool_w)

            if self.pool == 'max':
                self.arg_max = np.argmax(np.abs(self.pool_x_col), axis=1)
                self.pool_y = np.zeros(self.arg_max.shape, dtype='complex64')
                self.pool_y[np.arange(self.arg_max.shape[0])] = self.pool_x_col[np.arange(self.arg_max.shape[0]), self.arg_max[np.arange(self.arg_max.shape[0])]]
            elif self.pool == 'avg':
                self.pool_y = np.mean(self.pool_x_col, axis=1)
            self.pool_y = self.pool_y.reshape(batchsize, pool_out_h, pool_out_w, channel).transpose(0, 3, 1, 2)
            out = self.pool_y
        #print('b')
        return out

    def backward(self, t):
        if self.pool_or_not:
            # Pooling層の逆伝搬
            t = t.transpose(0,2,3,1)
            pool_size = self.pool_h * self.pool_w

            if self.pool == 'max':
                dmax = 2*np.ones((t.size, pool_size), dtype='complex64')
                dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = t.flatten()
                dmax = dmax.reshape(t.shape + (pool_size,))

                t1_col = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
                t1 = col2im(t1_col, self.conv_y.shape, self.pool_h, self.pool_w, self.pool_stride, self.pool_pad)
                t13 = t1
                t1[t1 == 2] = self.conv_y[t1 == 2]  # Convolutionの重みの教師信号
            elif self.pool == 'avg':
                dmax = np.zeros((t.size, pool_size), dtype='complex64')
                for i in range(pool_size):
                    dmax[np.arange(self.pool_x_col.shape[0]), i ] = 1 / self.pool_x_col.shape[1] * t.flatten()
                dmax = dmax.reshape(t.shape + (pool_size,))

                t1_col = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
                t1 = col2im(t1_col, self.conv_y.shape, self.pool_h, self.pool_w, self.pool_stride, self.pool_pad)
                t13 = t1

            #t1_col = t1.transpose(0,2,3,1)
            #t1_col = t1.reshape(t1_col.shape[0] * t1_col.shape[1] * t1_col.shape[2], -1)
            #t12 = col2im(t1_col, self.conv_y.shape, self.pool_h, self.pool_w, self.pool_stride, self.pool_pad)
            #t12[t12 == 0] = 0
            t12 = t1

        # Convolution層の逆伝搬
        filternum, channel, filter_h, filter_w = self.W.shape
        if not self.pool_or_not:
            t1 = t
            t12 = t

        t1_col2 = t12.transpose(0,2,3,1).reshape(-1, filternum)

        if self.batchnorm:
            t1_col2 = (np.sqrt(self.sigma_sq + self.epsilon) * np.abs(t1_col2)  + self.min - self.epsilon) * np.exp(1.j * np.angle(t1_col2))

        if backpro == 0:
            if cbp == 0:
                t_tmp = np.dot( self.mag*self.W_col, np.conj(t1_col2.T) )
            if cbp == 1:
                dmax = g_repmat(t.flatten().reshape(-1, 1), 1, pool_size)
                dmax = dmax.reshape(t.shape + (pool_size,))
                t1_col = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
                t1 = col2im(t1_col, self.conv_y.shape, self.pool_h, self.pool_w, self.pool_stride, self.pool_pad)
                t1_col22 = t1.transpose(0,2,3,1).reshape(-1, filternum)
                t_tmp = np.dot( self.W_col, np.conj(t1_col22.T) )
            if cbp == 2:
                t13_col = t13.transpose(0,2,3,1).reshape(-1, filternum)
                t_tmp = np.dot( self.W_col, np.conj(t13_col.T) )

            t0_col = activation( np.conj( t_tmp.T ) )
            t0 = col2im(t0_col, self.x.shape, filter_h, filter_w, self.conv_stride, self.conv_pad)  # 前層の教師信号



        # Wの更新
        if not self.memory_save:
            lis = []
            for i in range(self.x_col.shape[0]):
                lis.append([i]*self.W_col.shape[0])
            lis = [flatten for inner in lis for flatten in inner]
            y_b = g_repmat(self.conv_y_col, self.W_col.shape[0], 1)[lis].reshape((self.x_col.shape[0], self.W_col.shape[0], self.W_col.shape[1]))
            t_b = g_repmat(t1_col2, self.W_col.shape[0], 1)[lis].reshape((self.x_col.shape[0], self.W_col.shape[0], self.W_col.shape[1]))
            u_b = g_repmat(self.u_col, self.W_col.shape[0], 1)[lis].reshape((self.x_col.shape[0], self.W_col.shape[0], self.W_col.shape[1]))

            lis = []
            for i in range(self.x_col.shape[0]):
                lis.append([i]*self.W_col.shape[1])
            lis = [flatten for inner in lis for flatten in inner]
            x_b = g_repmat(self.x_col.T, 1, self.W_col.shape[1])[:,lis].T.reshape((self.x_col.shape[0], self.W_col.shape[1], self.W_col.shape[0])).transpose(0,2,1)
            theta_b = np.angle(y_b) - np.angle(x_b) - np.angle(self.W_col)

            self.dWa = self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.cos(theta_b) \
                                - np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-10) * np.sin(theta_b))
            self.dWp = self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.sin(theta_b) \
                                + np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-10) * np.cos(theta_b))
            self.dWa = np.sum(self.dWa, axis=0) #/ np.sqrt(self.x_col.shape[0])
            self.dWp = np.sum(self.dWp, axis=0) #/ np.sqrt(self.x_col.shape[0])
        else:
            self.dWa = np.zeros(self.W_col.shape, dtype='float32')
            self.dWp = np.zeros(self.W_col.shape, dtype='float32')

            for b in range(self.x_col.shape[0]):
                y_b = g_repmat(self.conv_y_col[b], self.W_col.shape[0], 1)
                t_b = g_repmat(t1_col2[b], self.W_col.shape[0], 1)
                u_b = g_repmat(self.u_col[b], self.W_col.shape[0], 1)
                x_b = g_repmat(self.x_col[b].reshape((self.x_col.shape[1], 1)), 1, self.W_col.shape[1])
                theta_b = np.angle(y_b) - np.angle(x_b) - np.angle(self.W_col)
                self.dWa += self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.cos(theta_b) \
                                    - np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-10) * np.sin(theta_b))
                self.dWp += self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.sin(theta_b) \
                                    + np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-10) * np.cos(theta_b))
            #self.dWa /= np.sqrt(self.x_col.shape[0])
            #self.dWp /= np.sqrt(self.x_col.shape[0])


        if self.bias:
            if not self.memory_save:
                y_b = self.conv_y_col.T
                t_b = t1_col2.T
                u_b = self.u_col.T
                x_b = np.ones((self.b.shape[0], y_b.shape[1]), dtype='complex64')
                b_b = self.b.reshape((self.b.shape[0], 1))
                theta_b = np.angle(y_b) - np.angle(b_b)
                self.dba = self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.cos(theta_b) \
                                    - np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-7) * np.sin(theta_b))
                self.dbp = self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.sin(theta_b) \
                                    + np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-7) * np.cos(theta_b))
                self.dba = np.sum(self.dba, axis=1)# / np.sqrt(self.dba.shape[0])
                self.dbp = np.sum(self.dbp, axis=1)# / np.sqrt(self.dba.shape[0])
            else:
                self.dba = np.zeros((self.b.shape[0], 1), dtype='float32')
                self.dbp = np.zeros((self.b.shape[0], 1), dtype='float32')

                for b in range(self.conv_y_col.shape[0]):
                    y_b = self.conv_y_col[b].reshape(self.conv_y_col[b].shape[0], 1)
                    t_b = t1_col2[b].reshape(t1_col2[b].shape[0], 1)
                    u_b = self.u_col[b].reshape(self.u_col[b].shape[0], 1)
                    x_b = g_repmat(1.+0.j, self.b.shape[0], 1)
                    b_b = self.b.reshape((self.b.shape[0], 1))
                    theta_b = np.angle(y_b) - np.angle(b_b)
                    self.dba += self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.cos(theta_b) \
                                        - np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-10) * np.sin(theta_b))
                    self.dbp += self.mag*((1 - np.abs(y_b)**2) * (np.abs(y_b) - np.abs(t_b)*np.cos(np.angle(y_b) - np.angle(t_b))) * np.abs(x_b) * np.sin(theta_b) \
                                        + np.abs(y_b) * np.abs(t_b) * np.sin(np.angle(y_b) - np.angle(t_b)) * np.abs(x_b) / (np.abs(u_b)+1e-10) * np.cos(theta_b))
                self.dba = self.dba.flatten() #/ np.sqrt(self.dba.shape[0])
                self.dbp = self.dbp.flatten()# / np.sqrt(self.dba.shape[0])


        if self.optimizer == 'Adam':
            if self.m is None:
                self.m, self.v = {}, {}
                self.m["dWa"] = np.zeros_like(self.dWa, dtype='float32')
                self.m["dWp"] = np.zeros_like(self.dWp, dtype='float32')
                self.v["dWa"] = np.zeros_like(self.dWa, dtype='float32')
                self.v["dWp"] = np.zeros_like(self.dWp, dtype='float32')
            self.iter += 1
            lr_t_a  = self.lr_a * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
            lr_t_p  = self.lr_p * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
            self.m["dWa"] += (1 - beta1) * (self.dWa - self.m["dWa"])
            self.m["dWp"] += (1 - beta1) * (self.dWp - self.m["dWp"])
            self.v["dWa"] += (1 - beta2) * (self.dWa**2 - self.v["dWa"])
            self.v["dWp"] += (1 - beta2) * (self.dWp**2 - self.v["dWp"])

            newWabs = np.abs(self.W_col) - lr_t_a * self.m["dWa"] / (np.sqrt(self.v["dWa"]) + 1e-7)
            newWphase = np.angle(self.W_col) - lr_t_p * self.m["dWp"] / (np.sqrt(self.v["dWp"]) + 1e-7)

            if self.bias:
                if self.bm is None:
                    self.bm, self.bv = {}, {}
                    self.bm["dba"] = np.zeros_like(self.dba, dtype='float32')
                    self.bm["dbp"] = np.zeros_like(self.dbp, dtype='float32')
                    self.bv["dba"] = np.zeros_like(self.dba, dtype='float32')
                    self.bv["dbp"] = np.zeros_like(self.dbp, dtype='float32')
                lr_t_a  = self.lr_a * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
                lr_t_p  = self.lr_p * np.sqrt(1.0 - beta2**self.iter) / (1.0 - beta1**self.iter)
                self.bm["dba"] += (1 - beta1) * (self.dba - self.bm["dba"])
                self.bm["dbp"] += (1 - beta1) * (self.dbp - self.bm["dbp"])
                self.bv["dba"] += (1 - beta2) * (self.dba**2 - self.bv["dba"])
                self.bv["dbp"] += (1 - beta2) * (self.dbp**2 - self.bv["dbp"])

                newbabs = np.abs(self.b) - (lr_t_a * self.bm["dba"] / (np.sqrt(self.bv["dba"]) + 1e-7)).flatten()
                newbphase = np.angle(self.b) - (lr_t_p * self.bm["dbp"] / (np.sqrt(self.bv["dbp"]) + 1e-7)).flatten()


        elif self.optimizer == 'SGD':
            newWabs = np.abs(self.W_col) - self.lr_a * self.dWa
            newWphase = np.angle(self.W_col) - self.lr_p * self.dWp
            if self.bias:
                newbabs = np.abs(self.b) - self.lr_a * self.dba
                newbphase = np.angle(self.b) - self.lr_p * self.dbp

        self.W_col = newWabs * np.exp(1.j * newWphase)
        self.W = self.W_col.transpose(1, 0).reshape(filternum, channel, filter_h, filter_w)
        if self.bias:
            self.b = newbabs * np.exp(1.j * newbphase)

        #print(str(np.abs(self.W[0,0,0,0])) + ' ' + str(np.angle(self.W[0,0,0,0])))
        if backpro == 1:
            t_tmp = np.dot( self.W_col, np.conj(t1_col2.T) )
            t0_col = activation( np.conj( t_tmp.T ) )
            t0 = col2im(t0_col, self.x.shape, filter_h, filter_w, self.conv_stride, self.conv_pad)
        return t0

    def return_W(self):
        return self.W

    def set_W(self, W):
        self.W = W

    def return_b(self):
        return self.b

    def set_b(self,b):
        self.b = b

class Conv:
    def __init__(self, W, conv_stride=1, conv_pad=0):
        self.W = W
        self.W_col = None   # im2colしたW
        self.conv_stride = conv_stride
        self.conv_pad = conv_pad

        self.x = None       # 入力
        self.x_col = None   # im2colしたx
        self.u = None       # Convolutionを取った後
        self.u_col = None   # im2colしたu
        self.conv_y = None  # self.uを活性化関数に通した後
        self.conv_y_col = None # im2colしたself.conv_y


    def forward(self, x):
        # Convolution
        filternum, channel, filter_h, filter_w = self.W.shape
        batchsize, channel, height, width = x.shape
        conv_out_h = 1 + int((height + 2*self.conv_pad - filter_h) / self.conv_stride)
        conv_out_w = 1 + int((width + 2*self.conv_pad - filter_w) / self.conv_stride)

        self.x = x
        self.x_col = im2col(x, filter_h, filter_w, self.conv_stride, self.conv_pad)
        self.W_col = self.W.reshape(filternum, -1).T

        self.u_col = np.dot(self.x_col, self.W_col)
        self.u = self.u_col.reshape(batchsize, conv_out_h, conv_out_w, -1).transpose(0, 3, 1, 2)

        out = self.u

        return out

class OutputLayer:
    def __init__(self):
        self.x = None
        self.original_x_shape = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        if len(self.original_x_shape) == 4:
            x = x.T
        self.x = x

        return self.x

    def backward(self, t): # この層の教師出力tから前層の教師出力t_0を計算
        if len(self.original_x_shape) == 4:
            t_0 = t.T
            t_0 = t_0.reshape(*self.original_x_shape)

        return t_0

class InputLayer:
    def forward(self, x):
        xs =x.shape
        y = x * np.exp(-1.j*np.angle(x[:,:,0,0])).reshape((xs[0],xs[1],1,1))

        return y

    def backward(self, t): # この層の教師出力tから前層の教師出力t_0を計算
        return t
