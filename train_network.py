# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
#import numpy as np
from np import *
from config import GPU
from util import *
import matplotlib.pyplot as plt
from network import Network
from trainer import Trainer
import pickle
import datetime
from classify_whole_img import *
from visualize_filter2 import *
from mnist import load_mnist

def train_network_1():
    start = datetime.datetime.today()

    data_file = "direction_xy_dataset_28x28ver2_21_conf0.0"
    #data_file = "direction_xy_dataset_28x28ver2_21_split2"                # 教師データ
    #data_file = "test_dataset_2_10x10"
    image = "FujiHakone_1140x1138_HH_ver2_rollxy_mean_5.npy"
    #image = "FujiHakone_1140x1138_HH_ver2_mean_5.npy"    # 区分対象の画像
    with open(data_file + ".pkl", 'rb') as f:
            dataset = pickle.load(f)

    (x_train, t_train), (x_test, t_test) = (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])

    '''
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False, one_hot_label=False)
    x_train = x_train.astype('complex64')
    x_test = x_test.astype('complex64')
    trm = numpy.random.choice(x_train.shape[0], 3000)
    tsm = numpy.random.choice(x_test.shape[0], 3000)
    x_train = x_train[trm]
    x_test = x_test[tsm]
    t_train = t_train[trm]
    t_test = t_test[tsm]
    '''

    if GPU:
        x_train, x_test = to_gpu(x_train), to_gpu(x_test)

    lr={'lr_a':0.01, 'lr_p':0.01}   # 全結合層の学習率
    c_lr={'lr_a':0.005, 'lr_p':0.005}   # 畳み込み層の学習率
    c_mean = 0.0                   # 畳み込み|W|の初期値の平均値
    c_std=0.02                     # 畳み込み|W|の初期値の標準偏差
    mean = 0.0                      # 全結合|W|の初期値の平均値
    std = 0.02                      # 全結合|W|の初期値の標準偏差　（以上４つは he=True としたとき使われない）
    epoch = 20                      # 更新のepoch数
    mini_batch = 32                 # ミニバッチに含む教師信号の個数
    pool_size = 2                  # プーリング窓のサイズ（縦＝横）
    pool_stride = 2                 # プーリングのストライド
    set_lr = False                  # 学習率を徐々に小さくする（通常はFalseにしておく）
    optimizer = 'SGD'              # 最適化手法('SGD'または'Adam')
    he = True                       # Wの振幅の初期値を　|（平均０，標準偏差sqrt( 2pi/(接続ニューロン数) )の正規分布 ）|　からとる（Heの初期値，Xavierの初期値のまね）
    mag=1.0                         # 活性化関数が tanh(|mag * u|) * exp(1.j * angle(u) ) になる　（magが実数の場合のみ対応）
    pool = 'max'
    bias = True
    input_layer = False
    classify_img = True
    params_load = False
    params = "result/201905231216/params.pkl"
    memory_save = False  # Falseにするとupdateで要素数の多い行列を使う


    network = Network(learning_rate=lr, c_learning_rate=c_lr,
                        c_weight_init_mean=c_mean, c_weight_init_std=c_std, weight_init_mean=mean, weight_init_std=std,
                        pool_size=pool_size, pool_stride=pool_stride,
                        optimizer=optimizer, he=he, mag=mag, pool=pool, bias=bias, input_layer=input_layer, memory_save=memory_save)
    win_ch, win_row, win_col = network.input_dim

    if params_load:
        network.load_params(file_name=params)    # Wの初期値をロード

    trainer = Trainer(network, x_train, t_train, x_test, t_test,
                      epochs=epoch, mini_batch_size=mini_batch,
                      evaluate_sample_num_per_epoch=100,
                      set_lr=set_lr)

    today = datetime.datetime.today()
    file_name = "log_" + today.strftime("%Y%m%d%H%M")
    path = "result/" + today.strftime("%Y%m%d%H%M") + "/"
    os.makedirs(path)



    trainer.train(file_name, path)

    learning_end = datetime.datetime.today()

    with open(path + "h_params" + file_name + ".txt", "a") as f:
        f.write("GPU : " + str(GPU) +'\n')
        f.write("params load : " + str(params_load) +'\n')
        if params_load:
            f.write("loaded params : " + params +'\n')
        f.write("data : " + data_file +'\n')
        f.write("image : " + image +'\n')
        f.write("x_shape = " + str(x_train.shape) +'\n')
        f.write("epochs = " + str(epoch) +'\n')
        if he:
            f.write("He init : " + str(he) +'\n')
        else:
            f.write("He init : " + str(he) +'\n')
            f.write("c_mean = " + str(c_mean) + '\n')
            f.write("c_std = " + str(c_std) + '\n')
            f.write("mean = " + str(mean) + '\n')
            f.write("std = " + str(std) + '\n')
        f.write("set lr =" + str(set_lr) + '\n')
        f.write("pool : " + pool +'\n')
        f.write("pool size = " + str(pool_size) + '\n')
        f.write("pool stride = " + str(pool_stride) + '\n')
        f.write("mini batch size = " + str(mini_batch) +'\n')
        f.write("lr_a : " + str(lr['lr_a']) + " lr_p : " + str(lr['lr_p']) +'\n')
        f.write("c_lr_a : " + str(c_lr['lr_a']) + " c_lr_p : " + str(c_lr['lr_p']) +'\n')
        f.write("activation function : tanh(|" + str(mag) + "*u|)*exp(j arg(u))\n")
        f.write("optimizer : " + optimizer +'\n')
        f.write("bias : " + str(bias) + '\n')
        f.write("input layer : " + str(input_layer) + '\n')
        f.write("network shape : ")
        for key in network.params.keys():
            if not (key[0] == 'b' and bias == False):
                f.write(key + str(network.params[key].shape) +'\n')
        f.write("class size = " + str(network.class_size) + '\n')
        f.write("teacher : \n")
        f.write(str(network.teacher_label()) + '\n')

    # Wの保存
    network.save_params(path + "params.pkl")
    print("Saved Network Parameters!")

    #print(np.angle(network.layers[0].W[0]))

    if classify_img:
        classify_ch(network=network, img=image, filepath=path, win_row=win_row, win_col=win_col, stride=1)  # 画像全体を区分

    classify_end = datetime.datetime.today()
    time1 = learning_end - start
    time2 = classify_end - start
    time1_minute = int(time1.days * 24 * 60 + time1.seconds / 60)
    time2_minute = int(time2.days * 24 * 60 + time2.seconds / 60)
    with open(path + "h_params" + file_name + ".txt", "a") as f:
        f.write("learning timedelta : " + str(time1_minute) + "min \n")
        f.write("start to finish timedelta : " + str(time2_minute) + "min \n")

    if not GPU:
        for i in range(network.conv_num):
            filter_show(network.layers[i].W, num=i+1, filepath=path)

def main():
    train_network_1()

if __name__ == '__main__':
    main()
