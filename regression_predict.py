# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
from dataset.california import load_california
# from dataset.boston import my_load_boston
from dataset.easy_data import load_easy_data, load_easy_data_sin
# from common.multi_layer_net_extend import MultiLayerNetExtend
from common.multi_layer_net_regression import MultiLayerNetRegression
from common.optimizer import SGD, Adam
from common.util import save_params, load_params
import time


# 未完成
if __name__ == "__main__":
    run = True
    x_train, x_test, y_train, y_test = load_california(debug=False)

    max_epochs = 10
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.01

    y_test = y_test.reshape((y_test.shape[0],1))

    if(run):
       
        #MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latiture, Longitude
        network = MultiLayerNetRegression(
            input_size=8, 
            hidden_size_list=[
            100,1000,100,
            ], 
            output_size=1,
            )
            
        params = load_params("test1")

        network.load_preset(params)

        guess = network.predict(x_test[500:550])
        plt.plot(guess, color="red", linestyle="--")

        answer = y_test[500:550]

        plt.plot(range(answer.shape[0]),answer, color='blue')
    
        plt.show()

