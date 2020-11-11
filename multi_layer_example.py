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
    # x_train, x_test, y_train, y_test = load_easy_data()
    # x_train, x_test, y_train, y_test = load_easy_data_sin(size=1000,range_pi=2*np.pi)
    x_train, x_test, y_train, y_test = load_california(debug=False)
    x_train_full = x_train[:]

    y_train_full = y_train[:]
    print(x_train.shape)
    # 学習データを削減
    x_train = x_train[:300]
    y_train = y_train[:300]

    max_epochs = 500

    train_size = x_train.shape[0]
    batch_size = 100
    
    learning_rate = 0.01
    y_train = y_train.reshape((y_train.shape[0],1))
    y_test = y_test.reshape((y_test.shape[0],1))
    print("x_train: ",x_train.shape)
    print("y train: ",y_train.shape)
    print("x_test: ",x_test.shape)
    print("y_test: ",y_test.shape)

    if(run):
       
        #MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latiture, Longitude
        network = MultiLayerNetRegression(
            input_size=8, 
            hidden_size_list=[
            100,1000,100,
            ], 
            output_size=1,
            )
        
        optimizer = Adam(lr=learning_rate)
        
        train_acc_list = []

        iter_per_epoch = max(train_size / batch_size, 1)

        epoch_cnt = 0

        for i in range(1000000000):
            batch_mask = np.random.choice(train_size, batch_size)

            x_batch = x_train[batch_mask]
            y_batch = y_train[batch_mask]

            grads = network.gradient(x_batch, y_batch)
            optimizer.update(network.params, grads)

            train_acc = network.accuracy(x_train, y_train)
            
            # print("TEST",train_acc)
            if i % iter_per_epoch == 0:
                train_acc = network.accuracy(x_train, y_train)
                train_acc_list.append(train_acc)

                print("epoch:" + str(epoch_cnt) + " | " + str(train_acc))
                epoch_cnt += 1
                if epoch_cnt >= max_epochs:
                    break

        
        params = network.params
        save_params(params, network.input_size, network.hidden_size_list, network.output_size, "test1")

        x = np.arange(max_epochs)       
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
        plt.show()

        guess = network.predict(x_test[500:550])
        plt.plot(guess, color="red", linestyle="--")

        answer = y_test[500:550]

        plt.plot(range(answer.shape[0]),answer, color='blue')
    
        plt.show()

