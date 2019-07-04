# coding: utf-8
#import numpy as np
from np import *
import matplotlib.pyplot as plt
from network import Network
import colorsys

def im_hsv_to_rgb(filters):
    FN, C, FH, FW = filters.shape
    out_filter = np.zeros((filters.shape + (3,)))


def filter_show(filters, num=0, margin=3, scale=10, filepath=''):
    """
    c.f. https://gist.github.com/aidiary/07d530d5e08011832b12#file-draw_weight-py
    """
    FN, C, FH, FW = filters.shape #filter_num, channel, height, width
    img_num = 0
    vmax=np.abs(filters).max()
    vmin=np.abs(filters).min()

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        for j in range(C):
            ax = fig.add_subplot(FN, C, img_num + 1, xticks=[], yticks=[])
            ax.imshow(np.abs(filters[i, j]), cmap=plt.cm.gray, interpolation='nearest',vmax=vmax,vmin=vmin)
            img_num += 1
    plt.savefig(filepath + "W" + str(num) + "amp.png")
    #plt.show()
    #plt.pause(1e-10)

    img_num2 = 0

    fig2 = plt.figure()
    fig2.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(FN):
        for j in range(C):
            ax = fig2.add_subplot(FN, C, img_num2 + 1, xticks=[], yticks=[])
            ax.imshow(np.angle(filters[i, j]), cmap=plt.cm.hsv, interpolation='nearest', vmin=-np.pi, vmax=np.pi)
            img_num2 += 1
    plt.savefig(filepath + "W" + str(num) + "phase.png")
    #plt.show()
    #plt.pause(1e-10)

def filter_show_rt(filters, ax, cmap):
    FN, C, FH, FW = filters.shape
    for i in range(FN):
        for j in range(C):
            ax[str(i) + '_' + str(j)].imshow(filters[i, j], cmap=cmap, interpolation='nearest')


def filter_show_ax(filters, fig, cmap, num=0, margin=3, scale=10, filepath=''):
    FN, C, FH, FW = filters.shape #filter_num, channel, height, width
    img_num = 0
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    ax = {}
    lines = {}

    for i in range(FN):
        for j in range(C):
            ax[str(i) + '_' + str(j)] = fig.add_subplot(FN, C, img_num + 1, xticks=[], yticks=[])
            lines[str(i) + '_' + str(j)] = ax[str(i) + '_' + str(j)].imshow(filters[i, j], cmap=cmap, interpolation='nearest')
            img_num += 1
    return ax, lines

def filter_set_data(filters, lines):
    FN, C, FH, FW = filters.shape #filter_num, channel, height, width
    for i in range(FN):
        for j in range(C):
            lines[str(i) + '_' + str(j)].set_data(filters[i, j])

def fig_ax(filters):
    FN, C, FH, FW = filters.shape #filter_num, channel, height, width
    fig, ax = plt.subplots(FN, C)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    plt.setp(ax, xticks=[], yticks=[])
    '''
    if C == 1:
        for i in range(FN):
            plt.sca(ax[i])
            plt.xticks([])
            plt.yticks([])
    else:
        for i in range(FN):
            for j in range(C):
                plt.sca(ax[i,j])
                plt.xticks([])
                plt.yticks([])
    '''
    return fig, ax

def lines(filters, ax, cmap, vmin=None, vmax=None):
    FN, C, FH, FW = filters.shape #filter_num, channel, height, width
    lines = {}
    if FN == 1 & C == 1:
        lines[str(0) + '_' + str(0)] = ax.imshow(filters[0,0], cmap=cmap,  vmin=vmin, vmax=vmax, interpolation='nearest')

    elif C == 1:
        for i in range(FN):
            for j in range(C):
                lines[str(i) + '_' + str(j)] = ax[i].imshow(filters[i,j], cmap=cmap,  vmin=vmin, vmax=vmax, interpolation='nearest')
    else:
        for i in range(FN):
            for j in range(C):
                lines[str(i) + '_' + str(j)] = ax[i,j].imshow(filters[i,j], cmap=cmap,  vmin=vmin, vmax=vmax, interpolation='nearest')
    return lines

def set_lines(filters, lines):
    FN, C, FH, FW = filters.shape
    for i in range(FN):
        for j in range(C):
            lines[str(i) + '_' + str(j)].set_data(filters[i,j])


if __name__ == '__main__':
    network = Network()

    # 学習後の重み
    network.load_params("params.pkl")


    filter_show(network.params['W1'])
    filter_show(network.params['W2'])
