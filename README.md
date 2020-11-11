### 稲葉ゼミ
---
# Neural Network Original Module for Regression
### Based on https://github.com/oreilly-japan/deep-learning-from-scratch

## 回帰ネットワーク
- **学習** ``common/multi_layer_net_regression.py``
``` python 
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from dataset.california import load_california
from common.multi_layer_net_regression import MultiLayerNetRegression
from common.optimizer import SGD, Adam

run = True
x_train, x_test, y_train, y_test = load_california(debug=False)
x_train_full = x_train[:]
y_train_full = y_train[:]

# 学習データを削減
x_train = x_train[:300]
y_train = y_train[:300]

max_epochs = 500
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
y_train = y_train.reshape((y_train.shape[0],1))
y_test = y_test.reshape((y_test.shape[0],1))

if(run):
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
    #学習

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
```
- **保存** ``multi_layer_example.py & common/util.py save_params``
```python 
from common.util import save_params
#学習
#...
#...
#...
#学習完了

#パラメータ・重み・バイアスをpickleファイルとして保存
params = network.params
now = datetime.now()

#作成時間をファイル名として保存(.pickleは省く)
save_params(
    params, 
    network.input_size, 
    network.hidden_size_list, 
    network.output_size, 
    now.strftime("%m-%d-%Y_%H-%M-%S")
)

```

- **予測** ``regression_predict.py & common/util.py load_params``
```python
import numpy as np
import matplotlib.pyplot as plt
from dataset.california import load_california
from common.multi_layer_net_regression import MultiLayerNetRegression
from common.util import load_params #学習済みパラメータを読み込む

# 学習済みのネットワークを使って予測をする

#データを読み込むが、今回はx_testとy_testのみ使う
x_train, x_test, y_train, y_test = load_california(debug=False)

y_test = y_test.reshape((y_test.shape[0],1))

#パラメータを読み込む
#FILE_NAME (.pickleは省く)
params_input = load_params("FILE_NAME")

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
```



## 追加ファイル

- **dataset/california.py**
    - 必須モジュール
        - numpy
        - scikit-learn
        - (matplotlib)
        - (pandas)
    - Args:
        - split : *bool, default: True*
            - ``split = True``の場合、訓練データ・テストデータを分割した状態で返す。
            - ``split = False``の場合、sklearnのbunchデータを返す。
        - debug : *bool, default: False*
            - ``debug = True``の場合、データの説明、データ、統計量、散布図を表示する。*(Pandas, matplotlibが必要)*
            - ``debug = False``の場合、データを返すだけ。
    - Returns:
        - (X_train, X_test, y_train, y_test) : *tuple*
            - 訓練データの説明変数、テストデータの説明変数、訓練データの目的変数、テストデータの目的変数
        - california_data : *sklearn.bunch*
    - example:
    ``` python
    from dataset.california import load_california
    from sklearn.neighbors import KNeighborsRegressor

    # 分割されたデータの読み込み
    X_train, X_test, y_train, y_test = load_california()

    reg = KNeighborsRegressor(n_neighbors = 3)
    reg.fit(X_train, y_train)

    print("Test set prediction: {}".format(reg.predict(X_test)))
    print("Test set R^2: {}".format(reg.score(X_test, y_test)))
    ```
        
