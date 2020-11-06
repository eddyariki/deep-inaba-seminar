# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import matplotlib.pyplot as plt
# from dataset.california import load_california
# from dataset.boston import my_load_boston
from dataset.easy_data import load_easy_data
# from common.multi_layer_net_extend import MultiLayerNetExtend
from common.multi_layer_net_regression import MultiLayerNetRegression
from common.optimizer import SGD, Adam
import time


# 未完成
if __name__ == "__main__":
    run = True
    x_train, x_test, y_train, y_test = load_easy_data()
    x_train_full = x_train[:]
    y_train_full = y_train[:]
    # 学習データを削減
    x_train = x_train
    y_train = y_train

    max_epochs = 200
    train_size = x_train.shape[0]
    batch_size = 20
    learning_rate = 0.02
    y_train = y_train.reshape((y_train.shape[0],1))
    y_test = y_test.reshape((y_test.shape[0],1))
    print("x_train: ",x_train.shape)
    print("y train: ",y_train.shape)
    print("x_test: ",x_test.shape)
    print("y_test: ",y_test.shape)

    if(run):
       
        #MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latiture, Longitude
        network = MultiLayerNetRegression(input_size=1, hidden_size_list=[100,100,100,100], output_size=1,
                                    weight_init_std=.05)
        optimizer = SGD(lr=learning_rate)
        
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
        
        x = np.arange(max_epochs)       
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
        plt.show()


        guess = network.predict(x_test)
        plt.plot(guess, color="red", linestyle="--")

        answer = y_test

        plt.plot(range(answer.shape[0]),answer, color='blue')
    
        plt.show()