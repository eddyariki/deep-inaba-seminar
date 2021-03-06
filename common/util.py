# coding: utf-8
import numpy as np
import time
import pickle 

def save_params(params, 
            input_size, 
            hidden_size_list, 
            output_size, 
            file_name="params_output"):
    """
    学習された重みとバイアスをpickleファイルに書き出す
    既に存在するファイルは上書きされる
    Parameters
    ----------
    params: 学習済み重みとバイアス
    inputs: 
    file_name: 書き出しのファイル名
    Returns
    -------
    None
    """
    params_output = {}
    params_output["input_size"] = input_size
    params_output["hidden_size_list"] = hidden_size_list, 
    params_output["output_size"] = output_size, 
    params_output["params"] = params
    try:
        with open(file_name+".pickle", "wb") as handle:
            pickle.dump(params_output, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved output to: " + file_name + ".pickle")
    except Exception as e:
        print(e)

def load_params(file_name="params_output"):
    """
    学習された重みとバイアスのpickleファイルを読み込む
    Parameters
    ----------
    file_name: pickleファイル名
    Returns
    -------
    params: 辞書型重みとバイアス
    """
    try:
        with open(file_name+".pickle", "rb") as handle:
            params = pickle.load(handle)
            print("Loaded: "+ file_name + ".pickle")
        return params
    except Exception as e:
        print(e)

def smooth_curve(x):
    """損失関数のグラフを滑らかにするために用いる

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """データセットのシャッフルを行う

    Parameters
    ----------
    x : 訓練データ
    t : 教師データ

    Returns
    -------
    x, t : シャッフルを行った訓練データと教師データ
    """
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    return (input_size + 2*pad - filter_size) / stride + 1


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
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

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

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]



def timer(function):
    def time_taken(*args):
        start = time.perf_counter()
        val = function(*args)
        end = time.perf_counter() - start
        print("Time it took: ", end)
        return val
    return time_taken