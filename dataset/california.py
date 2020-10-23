from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_california(split = True, debug = False):
    california_data = fetch_california_housing(data_home = "./")

    features = california_data["feature_names"]
    X = california_data["data"]
    y = california_data["target"]

    if debug:

        import pandas as pd
        import matplotlib.pyplot as plt

        print("ABOUT DATA")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print(california_data["DESCR"])
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")

        print("DATA")
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        X_df = pd.DataFrame(X, columns = features)
        y_df = pd.DataFrame(y, columns = ["HousePrice"])
        data = pd.concat([X_df, y_df], axis = 1)
        print("1. data")
        print(data)
        print("\n2. statistical description of data")
        print(data.describe())
        print("\n3. lack of data")
        print(data.isnull().sum())
        N = X.shape[1]
        fig = plt.figure()
        for i in range(N):
            ax = fig.add_subplot((N+2)//3, 3, i + 1)
            ax.scatter(X[:, i], y, marker = ".")
            ax.set_xlabel(features[i])
            ax.set_ylabel("HousePrice")
        plt.show()
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n\n")

    if split:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

        return X_train, X_test, y_train, y_test
    
    return california_data

if __name__ == "__main__":
    load_california(debug = True)