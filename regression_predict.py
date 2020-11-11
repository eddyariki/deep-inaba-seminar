# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from dataset.california import load_california
from common.multi_layer_net_regression import MultiLayerNetRegression
from common.util import load_params #学習済みパラメータを読み込む

# 学習済みのネットワークを使って予測をする
if __name__ == "__main__":
    #データを読み込むが、今回はx_testとy_testのみ使う
    x_train, x_test, y_train, y_test = load_california(debug=False)

    y_test = y_test.reshape((y_test.shape[0],1))

    #パラメータを読み込む
    params_input = load_params("11-11-2020_14-10-55")

    #データ確認
    print(params_input["input_size"],params_input["hidden_size_list"],params_input["output_size"])

    #ネットワークを学習済みのデータで初期化
    network = MultiLayerNetRegression(
        input_size=params_input["input_size"], 
        hidden_size_list=params_input["hidden_size_list"][0], 
        output_size=params_input["output_size"][0],
        preset_params=params_input["params"]
        )
    #予測
    guess = network.predict(x_test[500:550])
    
    #精度確認
    acc = network.accuracy(x_test[500:550], y_test[500:550])
    print("Accuracy: ",acc)

    #プロット
    plt.plot(guess, color="red", linestyle="--")

    answer = y_test[500:550]

    plt.plot(range(answer.shape[0]),answer, color='blue')

    plt.show()

