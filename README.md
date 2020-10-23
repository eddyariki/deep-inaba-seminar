### 稲葉ゼミ
---
# Neural Network Original Module for Regression

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
    from dataset/california import load_california
    from sklearn.neighbors import KNeighborsRegressor

    # 分割されたデータの読み込み
    X_train, X_test, y_train, y_test = load_california()

    reg = KNeighborsRegressor(n_neighbors = 3)
    reg.fit(X_train, y_train)

    print("Test set prediction: {}".format(reg.predict(X_test)))
    print("Test set R^2: {}".format(reg.score(X_test, y_test)))
    ```
        
