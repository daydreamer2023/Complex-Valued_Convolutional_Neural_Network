# coding: utf-8
#import numpy as np
import numpy
import matplotlib.pyplot as plt
from network import Network
import scipy.misc
import sys
import pickle
from functions import *
from util import *
from config import GPU
import time

colormap = plt.cm.rainbow

def classify_image(network, image, output_size, win_row=28, win_col=28, stride=3):
    row, col = image.shape
    out_row = int(numpy.floor((row - win_row) / stride) + 1)
    out_col = int(numpy.floor((col - win_col) / stride) + 1)

    output_img = numpy.empty((out_row, out_col), dtype='int')

    max_iter = int(out_row*out_col)
    now_iter = 0

    T = network.teacher_label()

    for r in range(out_row):
        for c in range(out_col):
            x = r*stride
            y = c*stride
            input_win = image[x:x+win_row, y:y+win_col].reshape(1,1,win_row,win_col)
            output_img[r,c] = network.classify(input_win, T)
            now_iter += 1
            sys.stdout.write("\r %d / %d" %(now_iter, max_iter))
            sys.stdout.flush()

    return output_img

def classify_image_ch(network, image, output_size, win_row, win_col, stride=3):
    if image.ndim == 2:
        image = image.reshape(1, image.shape[0], image.shape[1])
    channel, row, col = image.shape
    out_row = int(numpy.floor((row - win_row) / stride) + 1)
    out_col = int(numpy.floor((col - win_col) / stride) + 1)

    output_img = numpy.empty((out_row, out_col), dtype='int')

    max_iter = int(out_row*out_col)
    now_iter = 0

    T = network.teacher_label()

    oomuro_list = numpy.zeros((1,2), dtype=int)
    first_flg = True

    for r in range(out_row):
        for c in range(out_col):
            x = r*stride
            y = c*stride
            input_win = image[:,x:x+win_row, y:y+win_col].reshape(1,channel,win_row,win_col)
            if GPU:
                input_win = to_gpu(input_win)
            win_class = network.classify(input_win, T)
            output_img[r,c] = win_class
            if win_class == 6:
                if first_flg:
                    oomuro_list[0] = [x, y]
                    first_flg = False
                else:
                    oomuro_list = numpy.concatenate((oomuro_list, ([[x,y]])), axis=0)
            now_iter += 1
            sys.stdout.write("\r %d / %d" %(now_iter, max_iter))
            sys.stdout.flush()

    return output_img, oomuro_list


def classify_image_ch_p(network, image, output_size, win_row, win_col, stride=3, psize=32):
    #under constraction
    if image.ndim == 2:
        image = image.reshape(1, image.shape[0], image.shape[1])
    channel, row, col = image.shape
    out_row = int(numpy.floor((row - win_row) / stride) + 1)
    out_col = int(numpy.floor((col - win_col) / stride) + 1)
    osize = out_row * out_col
    piter = int(osize / psize)
    rem = osize - psize*piter

    output_img = numpy.empty((out_row*out_col, ), dtype='int')

    now_iter = 0

    T = network.teacher_label()

    oomuro_list = numpy.zeros((1,2), dtype=int)
    first_flg = True

    for i in range(piter):
        input_wins = numpy.zeros((psize, channel, win_row, win_col), dtype='complex64')
        for p in range(psize):
            curwin = i*psize + p
            x = int(curwin/out_col)
            y = int(curwin - x*out_col)
            x*=stride
            y*=stride
            #print('%d, %d' %(x,y))
            input_wins[p] = image[:,x:x+win_row, y:y+win_col]
        #print(input_wins.shape)
        if GPU:
            input_wins = to_gpu(input_wins)
        wins_class = network.classify(input_wins, T)
        output_img[i*psize:(i+1)*psize] = wins_class
        now_iter += 1
        sys.stdout.write("\r %d / %d" %(now_iter, piter+1))
        #time.sleep(1)
        #sys.stdout.write("\r %d,  %d, %d,  %d, %d,  %d, %d" %(wins_class[wins_class==0].size,wins_class[wins_class==1].size,wins_class[wins_class==2].size,wins_class[wins_class==3].size,wins_class[wins_class==4].size,wins_class[wins_class==5].size,wins_class[wins_class==6].size))
        sys.stdout.flush()

    input_wins = numpy.zeros((rem, channel, win_row, win_col), dtype='complex64')
    for p in range(rem):
        curwin = piter*psize + p
        x = int(curwin/out_col)
        y = int(curwin - x*out_col)
        x*=stride
        y*=stride
        #print('%d, %d' %(x,y))
        input_wins[p] = image[:,x:x+win_row, y:y+win_col]
    if GPU:
        input_wins = to_gpu(input_wins)
    wins_class = network.classify(input_wins, T)
    output_img[piter*psize:] = wins_class
    now_iter += 1
    sys.stdout.write("\r %d / %d" %(now_iter, piter+1))

    sys.stdout.flush()

    output_img = output_img.reshape((out_row, out_col))

    return output_img, oomuro_list


def classify(network, img, filepath, win_row=28, win_col=28):
    print('classifying...')
    #image = numpy.load('FujiHakone_mini.npy')
    '''
    data_file = "sea_other_dataset_comp"
    with open(data_file + ".pkl", 'rb') as f:
            dataset = pickle.load(f)

    (x_train, t_train), (x_test, t_test) = (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])
    x = x_train[100:200].transpose(2,0,1,3).reshape((28, -1))
    t = t_train[100:200]
    print(t)
    print(t.shape)
    t = to_one_hot_label(t, network.out_size)
    print(network.loss(x_train[100:200], t))
    '''
    image = numpy.load(img)
    output_img = classify_image(network, image, network.out_size)


    #scipy.misc.imsave(filepath + 'output.png', output_img)
    numpy.save(filepath + 'output.npy', output_img)
    plt.imsave(filepath + 'output.png', output_img, cmap=colormap)


def classify_ch(network, img, filepath, win_row=28, win_col=28, stride=3):
    print('classifying...')

    image = numpy.load(img)
    output_img, oomuro_list = classify_image_ch_p(network, image, network.out_size, win_row=win_row, win_col=win_col, stride=stride)
    numpy.save(filepath + 'output.npy', output_img)
    plt.imsave(filepath + 'output.png', output_img, cmap=colormap)
    numpy.save(filepath + 'oomuro_list.npy', oomuro_list)

if __name__ == '__main__':
    network = Network()
    network.load_params("params.pkl")
    classify(network)
